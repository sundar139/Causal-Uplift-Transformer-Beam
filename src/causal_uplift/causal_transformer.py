from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from causal_uplift.baselines import UpliftPrediction
from causal_uplift.causal_losses import (
    compute_positive_class_weights,
    factual_two_head_bce_loss,
    propensity_bce_loss,
)
from causal_uplift.evaluate import compute_uplift_metrics
from causal_uplift.transformer import NumericFeatureTokenizer


@dataclass(slots=True)
class CausalFTPrediction:
    treatment_probability: np.ndarray
    control_probability: np.ndarray
    uplift: np.ndarray

    def as_uplift_prediction(self) -> UpliftPrediction:
        return UpliftPrediction(
            treatment_proba=self.treatment_probability,
            control_proba=self.control_probability,
            uplift=self.uplift,
        )


class CausalFTTransformerUpliftModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.15,
        hidden_dim: int = 128,
        use_layer_norm: bool = True,
        use_propensity_head: bool = True,
    ) -> None:
        super().__init__()
        self.use_propensity_head = use_propensity_head
        self.tokenizer = NumericFeatureTokenizer(num_features=num_features, d_token=embedding_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embedding_dim) if use_layer_norm else nn.Identity()
        self.control_head = nn.Linear(embedding_dim, 1)
        self.treatment_head = nn.Linear(embedding_dim, 1)
        self.propensity_head = nn.Linear(embedding_dim, 1) if use_propensity_head else None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer(x)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        encoded = self.encoder(torch.cat([cls_token, tokens], dim=1))
        pooled = self.norm(encoded[:, 0])
        output = {
            "control_logit": self.control_head(pooled).squeeze(-1),
            "treatment_logit": self.treatment_head(pooled).squeeze(-1),
        }
        if self.propensity_head is not None:
            output["propensity_logit"] = self.propensity_head(pooled).squeeze(-1)
        return output


class CausalFTTransformerTrainer:
    def __init__(
        self,
        model: CausalFTTransformerUpliftModel,
        *,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 2048,
        epochs: int = 16,
        patience: int = 4,
        num_workers: int = 0,
        random_state: int = 42,
        factual_loss_weight: float = 1.0,
        propensity_loss_weight: float = 0.05,
        group_balance_weight: float = 0.20,
        positive_class_weighting: bool = True,
        checkpoint_metric: str = "qini_auc",
        log_mlflow_epochs: bool = True,
    ) -> None:
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.num_workers = num_workers
        self.factual_loss_weight = factual_loss_weight
        self.propensity_loss_weight = propensity_loss_weight
        self.group_balance_weight = group_balance_weight
        self.positive_class_weighting = positive_class_weighting
        self.checkpoint_metric = checkpoint_metric
        self.log_mlflow_epochs = log_mlflow_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)

        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.best_validation_metrics: dict[str, float] = {}
        self.history: list[dict[str, float]] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        treatment_train: np.ndarray,
        X_validation: np.ndarray,
        y_validation: np.ndarray,
        treatment_validation: np.ndarray,
        checkpoint_path: Path,
    ) -> dict[str, float]:
        train_dataset = TensorDataset(
            torch.from_numpy(X_train.astype(np.float32)),
            torch.from_numpy(y_train.astype(np.float32)),
            torch.from_numpy(treatment_train.astype(np.float32)),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        positive_weight = None
        if self.positive_class_weighting:
            positive_weight = compute_positive_class_weights(
                torch.from_numpy(y_train.astype(np.float32))
            ).to(self.device)

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        best_score: tuple[float, float] | None = None
        patience_counter = 0

        for epoch in tqdm(range(1, self.epochs + 1), desc="Training causal FT", leave=False):
            train_losses = self._train_epoch(train_loader, positive_weight)
            validation_prediction = self.predict(X_validation)
            validation_metrics = compute_uplift_metrics(
                y_true=y_validation,
                treatment=treatment_validation,
                uplift=validation_prediction.uplift,
                treatment_proba=validation_prediction.treatment_probability,
            )
            epoch_payload = {
                "epoch": float(epoch),
                **train_losses,
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
            }
            self.history.append(epoch_payload)
            if self.log_mlflow_epochs and mlflow.active_run() is not None:
                mlflow.log_metrics(
                    {key: value for key, value in epoch_payload.items() if key != "epoch"},
                    step=epoch,
                )

            current_score = (
                validation_metrics.get(self.checkpoint_metric, validation_metrics["qini_auc"]),
                validation_metrics["policy_gain_top20"],
            )
            if best_score is None or current_score > best_score:
                best_score = current_score
                patience_counter = 0
                self.best_validation_metrics = validation_metrics
                torch.save(self.model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        return self.best_validation_metrics

    def _train_epoch(
        self,
        loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        positive_weight: torch.Tensor | None,
    ) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {}
        batches = 0
        for features, labels, treatments in loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            treatments = treatments.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(features)
            factual_loss, factual_components = factual_two_head_bce_loss(
                outputs["control_logit"],
                outputs["treatment_logit"],
                labels,
                treatments,
                group_balance_weight=self.group_balance_weight,
                positive_class_weight=positive_weight,
            )
            total_loss = self.factual_loss_weight * factual_loss
            components = dict(factual_components)
            if "propensity_logit" in outputs:
                prop_loss, prop_components = propensity_bce_loss(
                    outputs["propensity_logit"],
                    treatments,
                    group_balance_weight=self.group_balance_weight,
                )
                total_loss = total_loss + self.propensity_loss_weight * prop_loss
                components.update(prop_components)
            components["total_loss"] = float(total_loss.detach().cpu().item())
            total_loss.backward()
            self.optimizer.step()

            batches += 1
            for key, value in components.items():
                totals[key] = totals.get(key, 0.0) + value

        return {key: value / max(batches, 1) for key, value in totals.items()}

    def predict(self, X: np.ndarray) -> CausalFTPrediction:
        self.model.eval()
        dataset = TensorDataset(torch.from_numpy(X.astype(np.float32)))
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        treatment_chunks: list[np.ndarray] = []
        control_chunks: list[np.ndarray] = []
        with torch.no_grad():
            for (features,) in loader:
                outputs = self.model(features.to(self.device))
                treatment_chunks.append(
                    torch.sigmoid(outputs["treatment_logit"]).cpu().numpy().astype(np.float64)
                )
                control_chunks.append(
                    torch.sigmoid(outputs["control_logit"]).cpu().numpy().astype(np.float64)
                )
        treatment_probability = (
            np.concatenate(treatment_chunks, axis=0)
            if treatment_chunks
            else np.array([], dtype=np.float64)
        )
        control_probability = (
            np.concatenate(control_chunks, axis=0)
            if control_chunks
            else np.array([], dtype=np.float64)
        )
        return CausalFTPrediction(
            treatment_probability=treatment_probability,
            control_probability=control_probability,
            uplift=treatment_probability - control_probability,
        )
