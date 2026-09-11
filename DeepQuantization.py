"""
DeepQuantization.py

Confidence-driven online deep classifier for IoT intrusion detection.

Revision highlights
-------------------
* Explicit switches for the adaptive feature gate, confidence-modulated
  learning rate, and online class weighting.
* Configurable class-weight update interval and smoothing constant.
* Numerically safe conversion of feature dictionaries to tensors.
* Correct confidence modulation over previously observed classes only.
* Separate full-precision online training and post-training int8 inference.
* Runtime diagnostics, component timing, parameter counts, and model-size
  reporting to support reproducible edge profiling.
* State-reset helpers for controlled drift experiments.

The class follows River's predict-one / learn-one interface. The intended
prequential sequence is:

    y_hat = model.predict_one(x)
    metric.update(y, y_hat)
    model.learn_one(x, y)

Dynamic quantization is post-training. Once quantize_model() has been called,
prediction uses the quantized inference copy, while learn_one() continues to
update the full-precision model. Call clear_quantized_model() after further
training before regenerating a quantized copy.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
from typing import Any, Deque, Dict, Iterable, Mapping, Optional
import copy
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
try:
    from river import base
except ImportError:  # Allows core-model smoke tests before River is installed.
    class _FallbackClassifier:
        _supervised = True
    class _FallbackBase:
        Classifier = _FallbackClassifier
    base = _FallbackBase()

try:
    torch.backends.quantized.engine = "qnnpack"
except Exception:
    pass


@dataclass
class TimingDiagnostics:
    confidence_forward_seconds: float = 0.0
    training_forward_backward_seconds: float = 0.0
    prediction_seconds: float = 0.0
    class_weight_seconds: float = 0.0
    learn_calls: int = 0
    predict_calls: int = 0
    weight_updates: int = 0

    def reset(self) -> None:
        for field_name in self.__dataclass_fields__:
            if field_name.endswith("_seconds"):
                setattr(self, field_name, 0.0)
            else:
                setattr(self, field_name, 0)

    def per_call_ms(self) -> Dict[str, float]:
        learn_denom = max(self.learn_calls, 1)
        pred_denom = max(self.predict_calls, 1)
        return {
            "confidence_forward_ms": 1000.0 * self.confidence_forward_seconds / learn_denom,
            "training_update_ms": 1000.0 * self.training_forward_backward_seconds / learn_denom,
            "prediction_ms": 1000.0 * self.prediction_seconds / pred_denom,
            "class_weight_update_ms": (
                1000.0 * self.class_weight_seconds / max(self.weight_updates, 1)
            ),
        }


def set_global_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _to_finite_float(value: Any, *, feature_name: str, missing_value: float) -> float:
    """Convert a feature value to a finite float.

    Missing, non-numeric, and non-finite values are replaced with missing_value.
    The notebook performs explicit validation before streaming; this guard
    prevents silent NaN propagation if an unexpected value reaches the model.
    """
    if value is None:
        return float(missing_value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(missing_value)
    if not math.isfinite(numeric):
        return float(missing_value)
    return numeric


def dict_to_tensor(
    x: Mapping[str, Any],
    feature_index: Mapping[str, int],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    missing_value: float = 0.0,
) -> torch.Tensor:
    """Convert a feature dictionary to a tensor with shape (1, d)."""
    if device is None:
        device = torch.device("cpu")
    vec = torch.full(
        (len(feature_index),),
        fill_value=float(missing_value),
        dtype=dtype,
        device=device,
    )
    for feature, value in x.items():
        idx = feature_index.get(feature)
        if idx is not None:
            vec[idx] = _to_finite_float(
                value, feature_name=str(feature), missing_value=missing_value
            )
    return vec.unsqueeze(0)


class AdaptiveFeatureGate(nn.Module):
    """Input-conditioned feature gate z = x ⊙ sigmoid(W_g x + b_g)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, input_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def activation(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate_proj(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.activation(x)


def _activation(name: str) -> nn.Module:
    options = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "swish": nn.SiLU(),
        "silu": nn.SiLU(),
        "leaky_relu": nn.LeakyReLU(0.01),
    }
    if name not in options:
        raise ValueError(f"Unsupported activation: {name!r}")
    return options[name]


class DeepFeatureExtractor(nn.Module):
    """Three-layer MLP followed by a compact latent projection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str = "leaky_relu",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers = []
        current = input_dim
        for _ in range(3):
            layers.extend(
                [
                    nn.Linear(current, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    _activation(activation),
                    nn.Dropout(dropout),
                ]
            )
            current = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassifierHead(nn.Module):
    def __init__(self, input_dim: int, n_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class CombinedModel(nn.Module):
    """Export wrapper for gate, feature extractor, and classifier head."""

    def __init__(
        self,
        gate: nn.Module,
        extractor: nn.Module,
        head: nn.Module,
        *,
        use_feature_gate: bool,
        logits: bool = True,
    ) -> None:
        super().__init__()
        self.gate = gate
        self.extractor = extractor
        self.head = head
        self.use_feature_gate = use_feature_gate
        self.logits = logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_feature_gate:
            x = self.gate(x)
        output = self.head(self.extractor(x))
        return output if self.logits else torch.softmax(output, dim=1)


class DeepOnlineClassifierDQ(base.Classifier):
    """Confidence-driven online deep classifier with post-training quantization.

    Parameters
    ----------
    feature_index:
        Mapping from feature name to tensor position.
    input_dim:
        Number of numerical input features.
    hidden_dim:
        Width of each hidden layer.
    output_dim:
        Latent representation dimension.
    n_classes:
        Number of output classes.
    deep_lr:
        Base Adam learning rate.
    activation:
        Hidden-layer activation.
    dropout:
        Dropout probability.
    seed:
        Random seed.
    window_size:
        Per-class confidence-window length.
    lr_modulation_alpha:
        Exponent α in the confidence-to-learning-rate transformation.
    lr_floor:
        Minimum effective learning rate as a fraction of deep_lr.
    use_feature_gate:
        Enable the input-conditioned feature gate.
    use_lr_modulation:
        Enable confidence-modulated learning rate.
    use_class_weights:
        Enable cumulative inverse-frequency class weighting.
    class_weight_update_every:
        Number of observations between class-weight recomputations.
    class_weight_smoothing:
        Additive smoothing constant λ.
    max_grad_norm:
        Gradient clipping threshold. Use None to disable clipping.
    missing_value:
        Fallback value for unexpected missing or non-finite features.
    device:
        PyTorch device. Defaults to CPU.
    collect_timing:
        Collect component-level runtime diagnostics.
    """

    _supervised = True

    def __init__(
        self,
        feature_index: Dict[str, int],
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 32,
        n_classes: int = 2,
        deep_lr: float = 1e-3,
        activation: str = "leaky_relu",
        dropout: float = 0.2,
        seed: int = 42,
        window_size: int = 100,
        lr_modulation_alpha: float = 0.8,
        lr_floor: float = 0.01,
        use_feature_gate: bool = True,
        use_lr_modulation: bool = True,
        use_class_weights: bool = True,
        class_weight_update_every: int = 500,
        class_weight_smoothing: float = 1.0,
        max_grad_norm: Optional[float] = 1.0,
        missing_value: float = 0.0,
        device: Optional[str] = None,
        collect_timing: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 0.0 < lr_floor <= 1.0:
            raise ValueError("lr_floor must lie in (0, 1]")
        if class_weight_update_every <= 0:
            raise ValueError("class_weight_update_every must be positive")
        if class_weight_smoothing <= 0:
            raise ValueError("class_weight_smoothing must be positive")

        set_global_seed(seed)
        self.seed = int(seed)
        self.feature_index = dict(feature_index)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.n_classes = int(n_classes)
        self.deep_lr = float(deep_lr)
        self.activation_name = activation
        self.dropout = float(dropout)
        self.window_size = int(window_size)
        self.lr_modulation_alpha = float(lr_modulation_alpha)
        self.lr_floor = float(lr_floor)
        self.use_feature_gate = bool(use_feature_gate)
        self.use_lr_modulation = bool(use_lr_modulation)
        self.use_class_weights = bool(use_class_weights)
        self.class_weight_update_every = int(class_weight_update_every)
        self.class_weight_smoothing = float(class_weight_smoothing)
        self.max_grad_norm = max_grad_norm
        self.missing_value = float(missing_value)
        self.device = torch.device(device or "cpu")
        self.collect_timing = bool(collect_timing)

        self.feature_gate = AdaptiveFeatureGate(self.input_dim).to(self.device)
        self.feature_extractor = DeepFeatureExtractor(
            self.input_dim,
            self.hidden_dim,
            self.output_dim,
            self.activation_name,
            self.dropout,
        ).to(self.device)
        self.classifier_head = ClassifierHead(
            self.output_dim, self.n_classes
        ).to(self.device)

        # Legacy aliases retained for older notebooks.
        self.bloom_filter = self.feature_gate

        self.optimizer = self._new_optimizer()
        self.loss_fn = nn.CrossEntropyLoss()

        self.class_counts = torch.zeros(self.n_classes, dtype=torch.float32)
        self._steps_since_weight_update = 0
        self.confidence_window: Dict[int, Deque[float]] = {
            k: deque(maxlen=self.window_size) for k in range(self.n_classes)
        }

        self.quantized_model: Optional[Dict[str, nn.Module]] = None
        self.loss_history: list[float] = []
        self.lr_history: list[float] = []
        self.last_loss: Optional[float] = None
        self.total_samples_trained = 0
        self.timing = TimingDiagnostics()

    def _new_optimizer(self) -> optim.Optimizer:
        return optim.Adam(self._parameters(), lr=self.deep_lr)

    def _parameters(self) -> Iterable[nn.Parameter]:
        yield from self.feature_gate.parameters()
        yield from self.feature_extractor.parameters()
        yield from self.classifier_head.parameters()

    def _train_mode(self, enabled: bool) -> None:
        self.feature_gate.train(enabled)
        self.feature_extractor.train(enabled)
        self.classifier_head.train(enabled)

    def _tensor(self, x: Mapping[str, Any]) -> torch.Tensor:
        return dict_to_tensor(
            x,
            self.feature_index,
            device=self.device,
            missing_value=self.missing_value,
        )

    def _gate(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_gate(x) if self.use_feature_gate else x

    def _forward_full_precision(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier_head(self.feature_extractor(self._gate(x)))

    def _average_confidence(self) -> Dict[int, float]:
        return {
            k: (sum(values) / len(values) if values else 0.0)
            for k, values in self.confidence_window.items()
        }

    def _modulated_lr(self, avg_confidence: Mapping[int, float]) -> float:
        if not self.use_lr_modulation:
            return self.deep_lr
        active = [
            float(value)
            for k, value in avg_confidence.items()
            if len(self.confidence_window[int(k)]) > 0
        ]
        uncertainty = 1.0 - min(active) if active else 1.0
        scale = max(
            self.lr_floor,
            max(0.0, uncertainty) ** self.lr_modulation_alpha,
        )
        return self.deep_lr * scale

    def _set_optimizer_lr(self, value: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = float(value)

    def set_base_lr(self, value: float) -> None:
        """Change η_base while retaining confidence modulation."""
        if value <= 0:
            raise ValueError("Learning rate must be positive")
        self.deep_lr = float(value)
        self._set_optimizer_lr(self.deep_lr)

    # Backward-compatible name.
    set_lr = set_base_lr

    def _recompute_class_weights(self) -> None:
        if not self.use_class_weights:
            self.loss_fn = nn.CrossEntropyLoss()
            return
        total = float(self.class_counts.sum().item())
        if total <= 0:
            return
        smoothed = self.class_counts + self.class_weight_smoothing
        weights = total / (self.n_classes * smoothed)
        weights = weights * (self.n_classes / weights.sum())
        self.loss_fn = nn.CrossEntropyLoss(weight=weights.to(self.device))

    def learn_one(
        self, x: Dict[str, float], y: int
    ) -> "DeepOnlineClassifierDQ":
        y_label = int(y)
        if not 0 <= y_label < self.n_classes:
            raise ValueError(
                f"Label {y_label} is outside [0, {self.n_classes - 1}]"
            )
        x_tensor = self._tensor(x)
        target = torch.tensor([y_label], dtype=torch.long, device=self.device)

        start = time.perf_counter()
        self._train_mode(False)
        with torch.no_grad():
            probabilities = torch.softmax(
                self._forward_full_precision(x_tensor), dim=1
            )
            true_class_confidence = float(probabilities[0, y_label].item())
        if self.collect_timing:
            self.timing.confidence_forward_seconds += time.perf_counter() - start

        self.confidence_window[y_label].append(true_class_confidence)
        average_confidence = self._average_confidence()
        effective_lr = self._modulated_lr(average_confidence)
        self._set_optimizer_lr(effective_lr)
        self.lr_history.append(effective_lr)

        self.class_counts[y_label] += 1.0
        self._steps_since_weight_update += 1
        if (
            self.use_class_weights
            and self._steps_since_weight_update >= self.class_weight_update_every
        ):
            weight_start = time.perf_counter()
            self._recompute_class_weights()
            if self.collect_timing:
                self.timing.class_weight_seconds += (
                    time.perf_counter() - weight_start
                )
                self.timing.weight_updates += 1
            self._steps_since_weight_update = 0

        train_start = time.perf_counter()
        self._train_mode(True)
        logits = self._forward_full_precision(x_tensor)
        loss = self.loss_fn(logits, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self._parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.collect_timing:
            self.timing.training_forward_backward_seconds += (
                time.perf_counter() - train_start
            )
            self.timing.learn_calls += 1

        self.last_loss = float(loss.item())
        self.loss_history.append(self.last_loss)
        self.total_samples_trained += 1

        # A quantized copy is stale after any full-precision update.
        self.quantized_model = None
        return self

    def _predict_probabilities_full_precision(
        self, x_tensor: torch.Tensor
    ) -> torch.Tensor:
        self._train_mode(False)
        with torch.no_grad():
            return torch.softmax(
                self._forward_full_precision(x_tensor), dim=1
            ).flatten()

    def _predict_probabilities_quantized(
        self, x_tensor: torch.Tensor
    ) -> torch.Tensor:
        assert self.quantized_model is not None
        gate = self.quantized_model["feature_gate"]
        extractor = self.quantized_model["feature_extractor"]
        head = self.quantized_model["classifier_head"]
        gate.eval()
        extractor.eval()
        head.eval()
        with torch.no_grad():
            if self.use_feature_gate:
                x_tensor = gate(x_tensor)
            return torch.softmax(head(extractor(x_tensor)), dim=1).flatten()

    def predict_proba_one(self, x: Dict[str, float]) -> Dict[int, float]:
        start = time.perf_counter()
        x_tensor = self._tensor(x)
        if self.quantized_model is None:
            probabilities = self._predict_probabilities_full_precision(x_tensor)
        else:
            probabilities = self._predict_probabilities_quantized(x_tensor.cpu())
        if self.collect_timing:
            self.timing.prediction_seconds += time.perf_counter() - start
            self.timing.predict_calls += 1
        values = probabilities.detach().cpu().tolist()
        return {index: float(values[index]) for index in range(self.n_classes)}

    def predict_one(self, x: Dict[str, float]) -> int:
        probabilities = self.predict_proba_one(x)
        return max(probabilities, key=probabilities.get)

    def get_torch_model(self, logits: bool = True) -> nn.Module:
        return CombinedModel(
            self.feature_gate,
            self.feature_extractor,
            self.classifier_head,
            use_feature_gate=self.use_feature_gate,
            logits=logits,
        )

    def clear_quantized_model(self) -> None:
        self.quantized_model = None

    def quantize_model(self) -> Dict[str, nn.Module]:
        """Create an inference-only dynamic-int8 copy.

        The adaptive gate remains in float32. Linear layers in the feature
        extractor and classifier head are dynamically quantized.
        """
        gate_copy = copy.deepcopy(self.feature_gate).cpu().eval()
        extractor_copy = copy.deepcopy(self.feature_extractor).cpu().eval()
        head_copy = copy.deepcopy(self.classifier_head).cpu().eval()

        quantized_extractor = torch.ao.quantization.quantize_dynamic(
            extractor_copy, {nn.Linear}, dtype=torch.qint8
        )
        quantized_head = torch.ao.quantization.quantize_dynamic(
            head_copy, {nn.Linear}, dtype=torch.qint8
        )
        self.quantized_model = {
            "feature_gate": gate_copy,
            "feature_extractor": quantized_extractor,
            "classifier_head": quantized_head,
        }
        return self.quantized_model

    def reset_feature_gate(self) -> None:
        self.feature_gate.reset_parameters()

    def reset_confidence(self) -> None:
        self.confidence_window = {
            k: deque(maxlen=self.window_size) for k in range(self.n_classes)
        }

    def reset_class_statistics(self) -> None:
        self.class_counts.zero_()
        self._steps_since_weight_update = 0
        self.loss_fn = nn.CrossEntropyLoss()

    def reset_model(
        self,
        *,
        seed: Optional[int] = None,
        preserve_class_statistics: bool = False,
        preserve_confidence: bool = False,
    ) -> None:
        """Reinitialize trainable parameters for reset-based drift baselines."""
        seed = self.seed if seed is None else int(seed)
        set_global_seed(seed)
        self.feature_gate = AdaptiveFeatureGate(self.input_dim).to(self.device)
        self.feature_extractor = DeepFeatureExtractor(
            self.input_dim,
            self.hidden_dim,
            self.output_dim,
            self.activation_name,
            self.dropout,
        ).to(self.device)
        self.classifier_head = ClassifierHead(
            self.output_dim, self.n_classes
        ).to(self.device)
        self.bloom_filter = self.feature_gate
        self.optimizer = self._new_optimizer()
        self.quantized_model = None
        if not preserve_class_statistics:
            self.reset_class_statistics()
        if not preserve_confidence:
            self.reset_confidence()

    def parameter_count(self, trainable_only: bool = False) -> int:
        parameters = self._parameters()
        if trainable_only:
            return sum(p.numel() for p in parameters if p.requires_grad)
        return sum(p.numel() for p in parameters)

    @staticmethod
    def _module_storage_bytes(module: nn.Module) -> int:
        total = 0
        for value in module.state_dict().values():
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
        return total

    def model_size_kb(self, quantized: bool = False) -> float:
        if quantized:
            if self.quantized_model is None:
                raise RuntimeError("Call quantize_model() before requesting quantized size")
            modules = self.quantized_model.values()
        else:
            modules = (
                self.feature_gate,
                self.feature_extractor,
                self.classifier_head,
            )
        return sum(self._module_storage_bytes(m) for m in modules) / 1024.0

    def gate_activation_one(self, x: Mapping[str, Any]) -> np.ndarray:
        """Return the unmultiplied sigmoid gate vector for analysis."""
        self.feature_gate.eval()
        with torch.no_grad():
            activation = self.feature_gate.activation(self._tensor(x))
        return activation.detach().cpu().numpy().reshape(-1)

    def get_diagnostics(self) -> Dict[str, Any]:
        class_weights = None
        if getattr(self.loss_fn, "weight", None) is not None:
            class_weights = {
                index: float(value)
                for index, value in enumerate(
                    self.loss_fn.weight.detach().cpu().tolist()
                )
            }
        return {
            "configuration": {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "n_classes": self.n_classes,
                "deep_lr": self.deep_lr,
                "window_size": self.window_size,
                "lr_modulation_alpha": self.lr_modulation_alpha,
                "lr_floor": self.lr_floor,
                "use_feature_gate": self.use_feature_gate,
                "use_lr_modulation": self.use_lr_modulation,
                "use_class_weights": self.use_class_weights,
                "class_weight_update_every": self.class_weight_update_every,
                "class_weight_smoothing": self.class_weight_smoothing,
                "max_grad_norm": self.max_grad_norm,
                "seed": self.seed,
            },
            "average_confidence_per_class": self._average_confidence(),
            "current_lr": (
                self.lr_history[-1] if self.lr_history else self.deep_lr
            ),
            "last_loss": self.last_loss,
            "total_samples_trained": self.total_samples_trained,
            "quantized_copy_available": self.quantized_model is not None,
            "class_counts": {
                index: int(value)
                for index, value in enumerate(self.class_counts.tolist())
            },
            "class_weights": class_weights,
            "parameter_count": self.parameter_count(),
            "full_precision_size_kb": self.model_size_kb(False),
            "timing_totals": asdict(self.timing),
            "timing_per_call_ms": self.timing.per_call_ms(),
        }


# Backward-compatible aliases.
NeuralFeatureGate = AdaptiveFeatureGate
NeuralBloomFilterLayer = AdaptiveFeatureGate
