from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from causal_uplift.baselines import UpliftPrediction


class NumericFeatureTokenizer(nn.Module):
    def __init__(self, num_features: int, d_token: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_features, d_token))
        self.bias = nn.Parameter(torch.zeros(num_features, d_token))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FTTransformerUpliftModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_token: int = 32,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.tokenizer = NumericFeatureTokenizer(num_features=num_features, d_token=d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        feedforward_dim = hidden_dim if hidden_dim is not None else d_token * 4
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        sequence = torch.cat([cls_token, tokens], dim=1)
        encoded = self.encoder(sequence)
        pooled = self.norm(encoded[:, 0])
        logits = self.head(pooled).squeeze(-1)
        return logits


class TorchUpliftTrainer:
    def __init__(
        self,
        model: FTTransformerUpliftModel,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 1024,
        epochs: int = 6,
        patience: int = 2,
        random_state: int = 42,
    ) -> None:
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
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
        self.loss_fn = nn.BCEWithLogitsLoss()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_validation: np.ndarray,
        y_validation: np.ndarray,
        checkpoint_path: Path,
    ) -> None:
        train_dataset = TensorDataset(
            torch.from_numpy(X_train.astype(np.float32)),
            torch.from_numpy(y_train.astype(np.float32)),
        )
        validation_dataset = TensorDataset(
            torch.from_numpy(X_validation.astype(np.float32)),
            torch.from_numpy(y_validation.astype(np.float32)),
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        validation_loader = DataLoader(
            validation_dataset, batch_size=self.batch_size, shuffle=False
        )

        best_val_loss = float("inf")
        patience_counter = 0

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        for _epoch in tqdm(range(1, self.epochs + 1), desc="Training transformer", leave=False):
            self.model.train()
            for features, labels in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(features)
                loss = self.loss_fn(logits, labels)
                loss.backward()
                self.optimizer.step()

            val_loss = self._evaluate_loss(validation_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)

    def _evaluate_loss(self, loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]) -> float:
        self.model.eval()
        total_loss = 0.0
        total_count = 0
        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(features)
                loss = self.loss_fn(logits, labels)
                batch_size = labels.shape[0]
                total_loss += float(loss.item()) * batch_size
                total_count += batch_size

        return total_loss / max(total_count, 1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        dataset = TensorDataset(torch.from_numpy(X.astype(np.float32)))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        chunks: list[np.ndarray] = []

        with torch.no_grad():
            for (features,) in loader:
                features = features.to(self.device)
                logits = self.model(features)
                probabilities = torch.sigmoid(logits)
                chunks.append(probabilities.detach().cpu().numpy().astype(np.float64))

        if not chunks:
            return np.array([], dtype=np.float64)
        return np.concatenate(chunks, axis=0)

    def predict_uplift(self, X_base: np.ndarray) -> UpliftPrediction:
        treatment_col = np.ones((X_base.shape[0], 1), dtype=np.float32)
        control_col = np.zeros((X_base.shape[0], 1), dtype=np.float32)

        treatment_input = np.concatenate([X_base.astype(np.float32), treatment_col], axis=1)
        control_input = np.concatenate([X_base.astype(np.float32), control_col], axis=1)

        treatment_proba = self.predict_proba(treatment_input)
        control_proba = self.predict_proba(control_input)
        uplift = treatment_proba - control_proba

        return UpliftPrediction(
            treatment_proba=treatment_proba,
            control_proba=control_proba,
            uplift=uplift,
        )
