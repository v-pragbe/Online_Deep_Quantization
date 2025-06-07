from __future__ import annotations
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from river import base

# === Helper ===
def dict_to_tensor(x: Dict[str, float], feature_index: Dict[str, int]) -> torch.Tensor:
    vec = torch.zeros(len(feature_index))
    for feature, value in x.items():
        if feature in feature_index:
            vec[feature_index[feature]] = value
    return vec.unsqueeze(0)  # Shape: (1, D)

# === Deep Network Components ===
class DeepFeatureExtractor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str = 'gelu',
        dropout: float = 0.2
    ):
        super().__init__()
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'leaky_relu': nn.LeakyReLU(0.01)
        }
        act = activations.get(activation, nn.ReLU())

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act,
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act,
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act,
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ClassifierHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

# === Baseline: No Distillation, No Quantization ===
class DeepOnlineClassifier(base.Classifier):
    def __init__(
        self,
        feature_index: Dict[str, int],
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        deep_lr: float = 1e-3,
        activation: str = 'gelu',
        dropout: float = 0.2,
        seed: int = 42
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.feature_index = feature_index

        # feature extractor + head
        self.feature_extractor = DeepFeatureExtractor(
            input_dim, hidden_dim, output_dim, activation, dropout
        )
        self.classifier_head = ClassifierHead(output_dim)

        # optimizer & loss
        self.optimizer = optim.Adam(
            list(self.feature_extractor.parameters()) +
            list(self.classifier_head.parameters()),
            lr=deep_lr
        )
        self.loss_fn = nn.BCELoss()

    def predict_proba_one(self, x: Dict[str, float]) -> Dict[bool, float]:
        x_tensor = dict_to_tensor(x, self.feature_index)
        self.feature_extractor.eval()
        self.classifier_head.eval()
        with torch.no_grad():
            feats = self.feature_extractor(x_tensor)
            logits = self.classifier_head(feats)
            prob = torch.sigmoid(logits).item()
        return {False: 1 - prob, True: prob}

    def learn_one(self, x: Dict[str, float], y: Any) -> DeepOnlineClassifier:
        # prepare inputs
        y_label = 1.0 if int(y) == 1 else 0.0
        x_tensor = dict_to_tensor(x, self.feature_index)

        # forward
        self.feature_extractor.train()
        self.classifier_head.train()
        feats = self.feature_extractor(x_tensor)
        logits = self.classifier_head(feats)
        prob = torch.sigmoid(logits)

        # compute loss
        target = torch.tensor([[y_label]], dtype=torch.float32)
        loss = self.loss_fn(prob, target)

        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return self

    def model_size_kb(self) -> float:
        total_bytes = 0
        for module in (self.feature_extractor, self.classifier_head):
            for param in module.state_dict().values():
                total_bytes += param.numel() * param.element_size()
        return total_bytes / 1024
