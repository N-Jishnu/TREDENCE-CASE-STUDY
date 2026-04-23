"""Self-pruning neural network for the Tredence AI Engineering case study.

Implementation:
- Part 1: PrunableLinear custom layer with learnable gate scores
- Part 2: L1 sparsity regularization loss
- Part 3: CIFAR-10 training and evaluation pipeline
- Part 4: Hard pruning conversion step
- Part 5: Compression and FLOPs reduction metrics
- Part 6: Visualization plots
- Part 7: Auto-generated Markdown report
"""

import argparse
import copy
import math
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm


matplotlib.use("Agg")
import matplotlib.pyplot as plt


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    pin_memory: bool = True
    pruning_threshold: float = 1e-2
    grad_clip_max_norm: float = 5.0
    gate_lr_multiplier: float = 10.0
    lambda_values: List[float] = None

    def __post_init__(self) -> None:
        if self.gate_lr_multiplier <= 0:
            raise ValueError("gate_lr_multiplier must be > 0.")
        if self.lambda_values is None:
            self.lambda_values = [1e-4, 5e-4, 1e-3]


def set_reproducibility(seed: int = 42) -> None:
    """Set random seeds and deterministic flags for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_temperature(epoch: int, num_epochs: int) -> float:
    """Linear annealing from 1.0 to 0.1."""
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")
    return max(0.1, 1.0 - 0.9 * (epoch / num_epochs))


class PrunableLinear(nn.Module):
    """Linear layer with learnable gate scores for self-pruning.

    Each weight has a matching gate score. During the forward pass, gate values
    are computed as sigmoid(gate_scores / temperature) and multiplied with the
    weight tensor before applying F.linear.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive integers.")

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters from scratch (no nn.Linear wrapper)."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        nn.init.zeros_(self.gate_scores)

    @staticmethod
    def _validate_temperature(temperature: float) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")

    def get_gate_values(self, temperature: float = 1.0) -> torch.Tensor:
        """Return activated gates in (0, 1)."""
        self._validate_temperature(temperature)
        return torch.sigmoid(self.gate_scores / temperature)

    def get_sparsity(self, threshold: float = 1e-2, temperature: float = 1.0) -> float:
        """Return fraction of gates below threshold."""
        if threshold < 0:
            raise ValueError("threshold must be >= 0.")

        gates = self.get_gate_values(temperature=temperature).detach()
        return (gates < threshold).float().mean().item()

    def get_gate_stats(self, temperature: float = 1.0, threshold: float = 1e-2) -> Dict[str, float]:
        """Return gate distribution statistics."""
        if threshold < 0:
            raise ValueError("threshold must be >= 0.")

        gates = self.get_gate_values(temperature=temperature).detach()
        return {
            "mean": gates.mean().item(),
            "median": gates.median().item(),
            "std": gates.std().item(),
            "pct_pruned": (gates < threshold).float().mean().item() * 100.0,
            "pct_active": (gates > 0.5).float().mean().item() * 100.0,
        }

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        self._validate_temperature(temperature)
        gates = self.get_gate_values(temperature=temperature)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


def compute_sparsity_loss(model: nn.Module, temperature: float = 1.0) -> torch.Tensor:
    """Compute L1 sparsity loss as sum of all activated gates.

    This follows the case-study definition of L1 regularization over gate values.
    """
    layer_penalties = []
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            layer_penalties.append(module.get_gate_values(temperature=temperature).sum())

    if not layer_penalties:
        raise ValueError("Model contains no PrunableLinear layers.")

    return torch.stack(layer_penalties).sum()


class SelfPruningNet(nn.Module):
    """Four-layer MLP with prunable linear layers for CIFAR-10."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = PrunableLinear(3072, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(p=0.2)

        self.fc2 = PrunableLinear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(p=0.2)

        self.fc3 = PrunableLinear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.drop3 = nn.Dropout(p=0.1)

        self.fc4 = PrunableLinear(256, 10)

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        x = x.view(x.size(0), -1)

        x = self.fc1(x, temperature=temperature)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.drop1(x)

        x = self.fc2(x, temperature=temperature)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.drop2(x)

        x = self.fc3(x, temperature=temperature)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)
        x = self.drop3(x)

        return self.fc4(x, temperature=temperature)


def _get_prunable_layers(model: nn.Module) -> List[PrunableLinear]:
    return [module for module in model.modules() if isinstance(module, PrunableLinear)]


def compute_sparsity_level(
    model: nn.Module,
    threshold: float = 1e-2,
    temperature: float = 1.0,
) -> float:
    """Compute global sparsity percentage across all prunable layers."""
    layers = _get_prunable_layers(model)
    if not layers:
        raise ValueError("Model contains no PrunableLinear layers.")

    total_gates = 0
    total_pruned = 0
    for layer in layers:
        gates = layer.get_gate_values(temperature=temperature).detach()
        total_gates += gates.numel()
        total_pruned += (gates < threshold).sum().item()

    return 100.0 * total_pruned / max(total_gates, 1)


def get_per_layer_stats(
    model: nn.Module,
    threshold: float = 1e-2,
    temperature: float = 1.0,
) -> Dict[str, Dict[str, float]]:
    """Collect gate statistics for each prunable layer."""
    stats: Dict[str, Dict[str, float]] = {}
    layer_index = 1
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            key = f"layer_{layer_index}"
            stats[key] = module.get_gate_stats(temperature=temperature, threshold=threshold)
            layer_index += 1
    return stats


def compute_compression_metrics(
    model: nn.Module,
    threshold: float = 1e-2,
    temperature: float = 1.0,
) -> Dict[str, float]:
    """Compute parameter and FLOPs reduction metrics for prunable layers."""
    if threshold < 0:
        raise ValueError("threshold must be >= 0.")

    layers = _get_prunable_layers(model)
    if not layers:
        raise ValueError("Model contains no PrunableLinear layers.")

    total_weights = 0
    active_weights = 0
    total_flops = 0
    active_flops = 0

    for layer in layers:
        gates = layer.get_gate_values(temperature=temperature).detach()
        n_weights = gates.numel()
        n_active = int((gates >= threshold).sum().item())

        total_weights += n_weights
        active_weights += n_active

        out_features, in_features = layer.weight.shape
        total_flops += 2 * in_features * out_features
        active_flops += 2 * n_active

    pruned_weights = total_weights - active_weights

    return {
        "total_weights": float(total_weights),
        "active_weights": float(active_weights),
        "pruned_weights": float(pruned_weights),
        "sparsity_pct": 100.0 * (pruned_weights / max(total_weights, 1)),
        "compression_ratio": float(total_weights) / max(float(active_weights), 1.0),
        "total_flops": float(total_flops),
        "active_flops": float(active_flops),
        "flops_reduction_pct": 100.0 * (1.0 - (active_flops / max(total_flops, 1))),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    temperature: float = 1.0,
) -> float:
    """Return classification accuracy (%) on a data loader."""
    model.eval()
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images, temperature=temperature)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / max(total, 1)


def get_cifar10_loaders(config: TrainingConfig, data_dir: str = "./data") -> tuple[DataLoader, DataLoader]:
    """Create CIFAR-10 train/test data loaders."""
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    lambda_value: float,
    temperature: float,
    grad_clip_max_norm: float,
) -> Dict[str, float]:
    """Train model for one epoch and return aggregate metrics."""
    model.train()

    running_cls = 0.0
    running_sparse = 0.0
    running_total = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(data_loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images, temperature=temperature)
        cls_loss = criterion(logits, labels)
        sparsity_loss = compute_sparsity_loss(model, temperature=temperature)
        total_loss = cls_loss + lambda_value * sparsity_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
        optimizer.step()

        batch_size = labels.size(0)
        running_cls += cls_loss.item()
        running_sparse += sparsity_loss.item()
        running_total += total_loss.item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    num_batches = len(data_loader)
    return {
        "train_accuracy": 100.0 * correct / max(total, 1),
        "classification_loss": running_cls / max(num_batches, 1),
        "sparsity_loss": running_sparse / max(num_batches, 1),
        "total_loss": running_total / max(num_batches, 1),
    }


def apply_hard_pruning(
    model: nn.Module,
    threshold: float = 1e-2,
    temperature: float = 1.0,
) -> nn.Module:
    """Return a copy of model with hard-pruned weights.

    For each PrunableLinear layer:
    - gate < threshold: corresponding weight forced to exactly 0.0,
      gate score forced to a very negative value (-1e6)
    - gate >= threshold: weight and gate score preserved as-is
    """
    if threshold < 0:
        raise ValueError("threshold must be >= 0.")

    model_copy = copy.deepcopy(model)
    for module in model_copy.modules():
        if isinstance(module, PrunableLinear):
            with torch.no_grad():
                gates = module.get_gate_values(temperature=temperature)
                keep_mask = (gates >= threshold).to(module.weight.dtype)
                prune_mask = keep_mask == 0

                module.weight.mul_(keep_mask)
                module.gate_scores[prune_mask] = -1e6

    return model_copy


def run_part3_experiments(
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
) -> Dict[float, Dict[str, Any]]:
    """Run the Part 3 training/evaluation pipeline for all lambda values."""
    results: Dict[float, Dict[str, Any]] = {}

    for lambda_value in config.lambda_values:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = SelfPruningNet().to(device)

        gate_params = []
        base_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith("gate_scores"):
                gate_params.append(param)
            else:
                base_params.append(param)

        optimizer = Adam(
            [
                {
                    "params": base_params,
                    "lr": config.learning_rate,
                    "weight_decay": config.weight_decay,
                },
                {
                    "params": gate_params,
                    "lr": config.learning_rate * config.gate_lr_multiplier,
                    "weight_decay": 0.0,
                },
            ]
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
        criterion = nn.CrossEntropyLoss()

        history: List[Dict[str, Any]] = []
        best_test_accuracy = float("-inf")
        best_epoch_result: Dict[str, Any] = {}
        best_state_dict: Dict[str, torch.Tensor] = {}

        print(f"\n=== Training with lambda={lambda_value:.1e} ===")

        for epoch in range(config.epochs):
            temperature = get_temperature(epoch, config.epochs)
            train_metrics = train_one_epoch(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                lambda_value=lambda_value,
                temperature=temperature,
                grad_clip_max_norm=config.grad_clip_max_norm,
            )

            test_acc = evaluate(model, test_loader, device, temperature=temperature)
            sparsity_pct = compute_sparsity_level(
                model,
                threshold=config.pruning_threshold,
                temperature=temperature,
            )
            per_layer = get_per_layer_stats(
                model,
                threshold=config.pruning_threshold,
                temperature=temperature,
            )

            epoch_result = {
                "epoch": epoch + 1,
                "temperature": temperature,
                "test_accuracy": test_acc,
                "sparsity_percent": sparsity_pct,
                "per_layer_stats": per_layer,
                **train_metrics,
            }
            history.append(epoch_result)

            is_better = test_acc > best_test_accuracy
            if is_better:
                best_test_accuracy = test_acc
                best_epoch_result = copy.deepcopy(epoch_result)
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

            scheduler.step()
            print(
                f"Epoch {epoch + 1:02d}/{config.epochs} | "
                f"tau={temperature:.2f} | "
                f"train={train_metrics['train_accuracy']:.2f}% | "
                f"test={test_acc:.2f}% | "
                f"sparsity={sparsity_pct:.2f}%"
            )

        best_model = SelfPruningNet().to(device)
        best_model.load_state_dict(best_state_dict)

        hard_pruned_model = apply_hard_pruning(
            best_model,
            threshold=config.pruning_threshold,
            temperature=best_epoch_result["temperature"],
        )
        hard_pruned_accuracy = evaluate(
            hard_pruned_model,
            test_loader,
            device,
            temperature=best_epoch_result["temperature"],
        )
        soft_compression_metrics = compute_compression_metrics(
            best_model,
            threshold=config.pruning_threshold,
            temperature=best_epoch_result["temperature"],
        )
        hard_compression_metrics = compute_compression_metrics(
            hard_pruned_model,
            threshold=config.pruning_threshold,
            temperature=best_epoch_result["temperature"],
        )

        results[lambda_value] = {
            "model": best_model,
            "final_model": model,
            "hard_pruned_model": hard_pruned_model,
            "history": history,
            "best_epoch": best_epoch_result["epoch"],
            "best_temperature": best_epoch_result["temperature"],
            "best_test_accuracy": best_epoch_result["test_accuracy"],
            "best_sparsity_percent": best_epoch_result["sparsity_percent"],
            "best_per_layer_stats": best_epoch_result["per_layer_stats"],
            "final_test_accuracy": history[-1]["test_accuracy"],
            "final_sparsity_percent": history[-1]["sparsity_percent"],
            "hard_pruned_test_accuracy": hard_pruned_accuracy,
            "compression_metrics_soft": soft_compression_metrics,
            "compression_metrics_hard": hard_compression_metrics,
        }

    return results


def assert_sparsity_monotonicity(
    results: Dict[float, Dict[str, Any]],
    lambda_values: List[float],
) -> None:
    """Assert best-epoch sparsity strictly increases with increasing lambda values."""
    ordered_lambdas = sorted([lmb for lmb in lambda_values if lmb in results])
    if len(ordered_lambdas) < 2:
        return

    ordered_sparsities = [results[lmb]["best_sparsity_percent"] for lmb in ordered_lambdas]
    comparisons = list(zip(ordered_lambdas[:-1], ordered_lambdas[1:], ordered_sparsities[:-1], ordered_sparsities[1:]))

    for left_lmb, right_lmb, left_s, right_s in comparisons:
        assert left_s < right_s, (
            "Best-epoch sparsity monotonicity check failed: expected "
            f"sparsity({left_lmb:.1e}) < sparsity({right_lmb:.1e}), got "
            f"{left_s:.4f} >= {right_s:.4f}."
        )


def collect_all_gate_values(model: nn.Module, temperature: float = 1.0) -> np.ndarray:
    """Collect all gate values from every prunable layer as a 1D array."""
    layers = _get_prunable_layers(model)
    if not layers:
        raise ValueError("Model contains no PrunableLinear layers.")

    gate_chunks = []
    for layer in layers:
        gates = layer.get_gate_values(temperature=temperature).detach().cpu().numpy().reshape(-1)
        gate_chunks.append(gates)
    return np.concatenate(gate_chunks)


def _mean_gate_value_from_epoch(epoch_result: Dict[str, Any]) -> float:
    per_layer = epoch_result.get("per_layer_stats", {})
    if not per_layer:
        return float("nan")
    layer_means = [layer_stats["mean"] for layer_stats in per_layer.values()]
    return float(np.mean(layer_means))


def select_best_tradeoff_lambda(results: Dict[float, Dict[str, Any]]) -> float:
    """Select lambda balancing hard-pruned accuracy and sparsity."""
    lambda_values = sorted(results.keys())
    hard_acc = np.array([results[lmb]["hard_pruned_test_accuracy"] for lmb in lambda_values], dtype=float)
    sparsity = np.array([results[lmb]["best_sparsity_percent"] for lmb in lambda_values], dtype=float)

    acc_range = np.ptp(hard_acc)
    sparsity_range = np.ptp(sparsity)

    hard_acc_norm = (hard_acc - hard_acc.min()) / (acc_range if acc_range > 0 else 1.0)
    sparsity_norm = (sparsity - sparsity.min()) / (sparsity_range if sparsity_range > 0 else 1.0)

    score = 0.6 * hard_acc_norm + 0.4 * sparsity_norm
    best_idx = int(np.argmax(score))
    return lambda_values[best_idx]


def plot_gate_distribution(
    results: Dict[float, Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    """Save histogram of gate values for best trade-off model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    best_lambda = select_best_tradeoff_lambda(results)
    best_model = results[best_lambda]["model"]
    gates = collect_all_gate_values(best_model, temperature=0.1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(gates, bins=100, range=(0.0, 1.0), color="#33658A", alpha=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("Gate Value")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(f"Gate Distribution (best trade-off lambda={best_lambda:.1e})")
    ax.grid(alpha=0.2, linestyle="--")

    output_path = output_dir / "gate_distribution.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "best_lambda": best_lambda,
        "path": str(output_path),
    }


def plot_gate_evolution(
    results: Dict[float, Dict[str, Any]],
    lambda_values: List[float],
    output_dir: Path,
) -> str:
    """Save gate evolution plot with sparsity, mean gates, and temperature."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax_sparsity = plt.subplots(figsize=(12, 7))
    ax_secondary = ax_sparsity.twinx()

    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(lambda_values)))

    for color, lambda_value in zip(colors, lambda_values):
        history = results[lambda_value]["history"]
        epochs = [h["epoch"] for h in history]
        sparsity_series = [h["sparsity_percent"] for h in history]
        mean_gate_series = [_mean_gate_value_from_epoch(h) for h in history]

        ax_sparsity.plot(
            epochs,
            sparsity_series,
            color=color,
            linewidth=2.0,
            label=f"Sparsity lambda={lambda_value:.1e}",
        )
        ax_secondary.plot(
            epochs,
            mean_gate_series,
            color=color,
            linewidth=1.6,
            linestyle="--",
            alpha=0.85,
            label=f"Mean gate lambda={lambda_value:.1e}",
        )

    ref_lambda = lambda_values[0]
    ref_history = results[ref_lambda]["history"]
    ref_epochs = [h["epoch"] for h in ref_history]
    ref_temp = [h["temperature"] for h in ref_history]
    ax_secondary.plot(
        ref_epochs,
        ref_temp,
        color="#202020",
        linewidth=2.0,
        linestyle=":",
        label="Temperature",
    )

    ax_sparsity.set_xlabel("Epoch")
    ax_sparsity.set_ylabel("Sparsity (%)")
    ax_secondary.set_ylabel("Mean Gate Value / Temperature")
    ax_sparsity.set_title("Gate Evolution During Training")
    ax_sparsity.grid(alpha=0.2, linestyle="--")

    handles_a, labels_a = ax_sparsity.get_legend_handles_labels()
    handles_b, labels_b = ax_secondary.get_legend_handles_labels()
    ax_sparsity.legend(handles_a + handles_b, labels_a + labels_b, loc="center right", fontsize=8)

    output_path = output_dir / "gate_evolution.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_sparsity_accuracy_tradeoff(
    results: Dict[float, Dict[str, Any]],
    lambda_values: List[float],
    output_dir: Path,
) -> str:
    """Save sparsity-accuracy trade-off chart with compression annotations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(lambda_values))
    soft_acc = np.array([results[lmb]["best_test_accuracy"] for lmb in lambda_values])
    hard_acc = np.array([results[lmb]["hard_pruned_test_accuracy"] for lmb in lambda_values])
    sparsity = np.array([results[lmb]["best_sparsity_percent"] for lmb in lambda_values])
    compression = np.array(
        [results[lmb]["compression_metrics_hard"]["compression_ratio"] for lmb in lambda_values],
        dtype=float,
    )

    fig, ax_acc = plt.subplots(figsize=(11, 6))
    width = 0.34
    bars_soft = ax_acc.bar(x - width / 2, soft_acc, width=width, color="#FF8C42", label="Soft accuracy")
    bars_hard = ax_acc.bar(x + width / 2, hard_acc, width=width, color="#2A9D8F", label="Hard accuracy")

    ax_sparse = ax_acc.twinx()
    ax_sparse.plot(x, sparsity, color="#264653", marker="o", linewidth=2.0, label="Sparsity")

    for idx, bar in enumerate(bars_hard):
        ax_acc.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{compression[idx]:.1f}x",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#1f1f1f",
        )

    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels([f"{lmb:.0e}" for lmb in lambda_values])
    ax_acc.set_xlabel("Lambda")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_sparse.set_ylabel("Sparsity (%)")
    ax_acc.set_title("Sparsity vs Accuracy Trade-off")
    ax_acc.grid(axis="y", alpha=0.2, linestyle="--")

    handles_a, labels_a = ax_acc.get_legend_handles_labels()
    handles_b, labels_b = ax_sparse.get_legend_handles_labels()
    ax_acc.legend(handles_a + handles_b, labels_a + labels_b, loc="lower right")

    output_path = output_dir / "sparsity_accuracy_tradeoff.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_training_curves(
    results: Dict[float, Dict[str, Any]],
    lambda_values: List[float],
    output_dir: Path,
) -> str:
    """Save 2x2 training dynamics plot across lambda values."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axis_map = {
        "test_accuracy": (axes[0, 0], "Test Accuracy (%)"),
        "sparsity_percent": (axes[0, 1], "Sparsity (%)"),
        "classification_loss": (axes[1, 0], "Classification Loss"),
        "total_loss": (axes[1, 1], "Total Loss"),
    }

    colors = plt.cm.plasma(np.linspace(0.15, 0.9, len(lambda_values)))
    for color, lambda_value in zip(colors, lambda_values):
        history = results[lambda_value]["history"]
        epochs = [h["epoch"] for h in history]
        for metric_key, (ax, title) in axis_map.items():
            values = [h[metric_key] for h in history]
            ax.plot(epochs, values, color=color, linewidth=2.0, label=f"lambda={lambda_value:.1e}")
            ax.set_title(title)
            ax.grid(alpha=0.2, linestyle="--")
            ax.set_xlabel("Epoch")

    axes[0, 0].legend(loc="best", fontsize=9)
    fig.suptitle("Training Curves")

    output_path = output_dir / "training_curves.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def generate_part6_plots(
    results: Dict[float, Dict[str, Any]],
    lambda_values: List[float],
    output_dir: str,
) -> Dict[str, str]:
    """Generate and save all Part 6 visualization artifacts."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gate_distribution_meta = plot_gate_distribution(results=results, output_dir=out_dir)
    gate_evolution_path = plot_gate_evolution(results=results, lambda_values=lambda_values, output_dir=out_dir)
    tradeoff_path = plot_sparsity_accuracy_tradeoff(results=results, lambda_values=lambda_values, output_dir=out_dir)
    training_curves_path = plot_training_curves(results=results, lambda_values=lambda_values, output_dir=out_dir)

    return {
        "best_lambda": f"{gate_distribution_meta['best_lambda']:.1e}",
        "gate_distribution": gate_distribution_meta["path"],
        "gate_evolution": gate_evolution_path,
        "sparsity_accuracy_tradeoff": tradeoff_path,
        "training_curves": training_curves_path,
    }


def _sorted_layer_keys(per_layer_stats: Dict[str, Dict[str, float]]) -> List[str]:
    return sorted(per_layer_stats.keys(), key=lambda k: int(k.split("_")[-1]))


def generate_part7_report(
    results: Dict[float, Dict[str, Any]],
    lambda_values: List[float],
    plot_paths: Dict[str, str],
    output_dir: str,
    report_filename: str = "report.md",
) -> str:
    """Generate Markdown report with tables, analysis, and plot links."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / report_filename

    first_lambda = lambda_values[0]
    first_per_layer = results[first_lambda]["best_per_layer_stats"]
    layer_keys = _sorted_layer_keys(first_per_layer)

    best_lambda = select_best_tradeoff_lambda(results)
    best_result = results[best_lambda]

    lines: List[str] = []
    lines.append("# Self-Pruning Neural Network - Report")
    lines.append("")
    lines.append("## 1. Why L1 Penalty on Sigmoid Gates Encourages Sparsity")
    lines.append(
        "The sparsity term applies an L1 penalty to activated gates, so each gate receives a consistent "
        "downward pressure while still being bounded in (0, 1) by sigmoid. Gates tied to less useful "
        "connections usually cannot justify staying open through classification gradients, so they are "
        "pushed near zero. Temperature annealing further sharpens decisions late in training, making "
        "the final gate distribution more bimodal and easier to hard-prune."
    )
    lines.append("")
    lines.append("## 2. Results")
    lines.append("")
    lines.append("### Best vs Final Epoch Summary")
    lines.append(
        "| Lambda (λ) | Best Epoch | Best Test Acc (%) | Best Sparsity (%) | Final Test Acc (%) | Final Sparsity (%) |"
    )
    lines.append("|:---|---:|---:|---:|---:|---:|")
    for lambda_value in lambda_values:
        run_result = results[lambda_value]
        lines.append(
            f"| {lambda_value:.0e} | {run_result['best_epoch']} | {run_result['best_test_accuracy']:.2f} | "
            f"{run_result['best_sparsity_percent']:.2f} | {run_result['final_test_accuracy']:.2f} | "
            f"{run_result['final_sparsity_percent']:.2f} |"
        )

    lines.append("")
    lines.append("### Hard Pruning Results")
    lines.append("| Lambda (λ) | Soft Accuracy at Best Epoch (%) | Hard Accuracy (%) | Compression Ratio |")
    lines.append("|:---|:---:|:---:|:---:|")
    for lambda_value in lambda_values:
        run_result = results[lambda_value]
        hard_metrics = run_result["compression_metrics_hard"]
        lines.append(
            f"| {lambda_value:.0e} | {run_result['best_test_accuracy']:.2f} | "
            f"{run_result['hard_pruned_test_accuracy']:.2f} | {hard_metrics['compression_ratio']:.2f}x |"
        )

    lines.append("")
    lines.append("### Per-Layer Sparsity Breakdown")
    header_cells = ["Lambda (λ)"] + layer_keys
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|" + "|".join([":---"] + [":---:" for _ in layer_keys]) + "|")
    for lambda_value in lambda_values:
        run_result = results[lambda_value]
        per_layer = run_result["best_per_layer_stats"]
        row_values = [f"{lambda_value:.0e}"]
        for layer_key in layer_keys:
            row_values.append(f"{per_layer[layer_key]['pct_pruned']:.2f}%")
        lines.append("| " + " | ".join(row_values) + " |")

    lines.append("")
    lines.append("### Parameter Reduction Summary")
    lines.append(
        "| Lambda (λ) | Total Parameters | Active Parameters | Compression Ratio | FLOPs Reduction (%) |"
    )
    lines.append("|:---|---:|---:|---:|---:|")
    for lambda_value in lambda_values:
        hard_metrics = results[lambda_value]["compression_metrics_hard"]
        lines.append(
            f"| {lambda_value:.0e} | {int(hard_metrics['total_weights'])} | "
            f"{int(hard_metrics['active_weights'])} | {hard_metrics['compression_ratio']:.2f}x | "
            f"{hard_metrics['flops_reduction_pct']:.2f} |"
        )

    lines.append("")
    lines.append("### Analysis")
    lines.append(
        f"Best trade-off lambda from this run is **{best_lambda:.1e}**, with hard-pruned accuracy "
        f"**{best_result['hard_pruned_test_accuracy']:.2f}%** at best-epoch sparsity "
        f"**{best_result['best_sparsity_percent']:.2f}%**. "
        "As lambda increases, sparsity usually increases and compression improves, while accuracy may drop "
        "after a point due to stronger regularization pressure."
    )

    lines.append("")
    lines.append("## 3. Gate Distribution")
    lines.append(f"![Gate Distribution]({Path(plot_paths['gate_distribution']).name})")
    lines.append("")
    lines.append("## 4. Gate Evolution During Training")
    lines.append(f"![Gate Evolution]({Path(plot_paths['gate_evolution']).name})")
    lines.append("")
    lines.append("## 5. Training Dynamics")
    lines.append(f"![Training Curves]({Path(plot_paths['training_curves']).name})")
    lines.append("")
    lines.append("## 6. Sparsity vs Accuracy Trade-off")
    lines.append(f"![Trade-off]({Path(plot_paths['sparsity_accuracy_tradeoff']).name})")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(report_path)


def _build_dummy_results_for_reporting(
    lambda_values: List[float],
) -> Dict[float, Dict[str, Any]]:
    """Create synthetic results used by Part 6/Part 7 sanity checks."""
    dummy_results: Dict[float, Dict[str, Any]] = {}

    for idx, lambda_value in enumerate(lambda_values):
        model = nn.Sequential(PrunableLinear(16, 8), nn.ReLU(), PrunableLinear(8, 4))
        history = []
        for epoch in range(1, 6):
            temperature = get_temperature(epoch - 1, 5)
            sparsity = 15.0 * idx + epoch * 4.0
            history.append(
                {
                    "epoch": epoch,
                    "temperature": temperature,
                    "test_accuracy": 40.0 + idx * 4.5 + epoch * 0.6,
                    "sparsity_percent": sparsity,
                    "classification_loss": 2.0 - epoch * 0.2 + idx * 0.05,
                    "total_loss": 2.2 - epoch * 0.22 + idx * 0.06,
                    "per_layer_stats": {
                        "layer_1": {
                            "mean": max(0.05, 0.55 - epoch * 0.05 - idx * 0.03),
                            "pct_pruned": min(100.0, sparsity + 2.0),
                        },
                        "layer_2": {
                            "mean": max(0.05, 0.50 - epoch * 0.04 - idx * 0.03),
                            "pct_pruned": max(0.0, sparsity - 2.0),
                        },
                    },
                }
            )

        hard_model = apply_hard_pruning(model, threshold=1e-2, temperature=1.0)
        best_epoch_result = history[-2]
        dummy_results[lambda_value] = {
            "model": model,
            "final_model": model,
            "hard_pruned_model": hard_model,
            "history": history,
            "best_epoch": best_epoch_result["epoch"],
            "best_temperature": best_epoch_result["temperature"],
            "best_test_accuracy": best_epoch_result["test_accuracy"],
            "best_sparsity_percent": best_epoch_result["sparsity_percent"],
            "best_per_layer_stats": best_epoch_result["per_layer_stats"],
            "final_test_accuracy": history[-1]["test_accuracy"],
            "final_sparsity_percent": history[-1]["sparsity_percent"],
            "hard_pruned_test_accuracy": best_epoch_result["test_accuracy"] - 0.8,
            "compression_metrics_soft": compute_compression_metrics(
                model,
                threshold=1e-2,
                temperature=1.0,
            ),
            "compression_metrics_hard": compute_compression_metrics(
                hard_model,
                threshold=1e-2,
                temperature=1.0,
            ),
        }

    return dummy_results


def _part1_gradient_flow_sanity_check(device: torch.device) -> None:
    """Quick check that gradients flow to both weight and gate_scores."""
    layer = PrunableLinear(8, 4).to(device)
    x = torch.randn(16, 8, device=device)
    y = torch.randint(0, 4, (16,), device=device)

    logits = layer(x, temperature=1.0)
    loss = F.cross_entropy(logits, y)
    loss.backward()

    assert layer.weight.grad is not None, "weight.grad is None"
    assert layer.gate_scores.grad is not None, "gate_scores.grad is None"


def _part2_sparsity_loss_sanity_check(device: torch.device) -> None:
    """Check sparsity loss formulation and differentiability."""
    model = nn.Sequential(
        PrunableLinear(8, 6),
        nn.ReLU(),
        PrunableLinear(6, 4),
    ).to(device)

    temperature = 0.7
    sparsity_loss = compute_sparsity_loss(model, temperature=temperature)
    assert sparsity_loss.ndim == 0, "Sparsity loss must be a scalar tensor"

    layer1 = model[0]
    layer2 = model[2]
    expected = (
        layer1.get_gate_values(temperature=temperature).sum()
        + layer2.get_gate_values(temperature=temperature).sum()
    )
    assert torch.allclose(sparsity_loss, expected), "L1 gate-sum sparsity loss mismatch"

    x = torch.randn(10, 8, device=device)
    y = torch.randint(0, 4, (10,), device=device)
    logits = model(x)
    total_loss = F.cross_entropy(logits, y) + 1e-4 * sparsity_loss
    total_loss.backward()

    assert layer1.gate_scores.grad is not None, "layer1 gate_scores.grad is None"
    assert layer2.gate_scores.grad is not None, "layer2 gate_scores.grad is None"


def _part3_pipeline_sanity_check(device: torch.device) -> None:
    """Quick structural sanity check for Part 3 pipeline pieces."""
    model = SelfPruningNet().to(device)
    images = torch.randn(8, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (8,), device=device)

    logits = model(images, temperature=1.0)
    assert logits.shape == (8, 10), "Model output shape must be [batch, 10]"

    total_loss = F.cross_entropy(logits, labels) + 1e-4 * compute_sparsity_loss(model, temperature=1.0)
    total_loss.backward()

    prunable_layers = _get_prunable_layers(model)
    assert len(prunable_layers) == 4, "SelfPruningNet must have four PrunableLinear layers"
    for layer in prunable_layers:
        assert layer.weight.grad is not None, "Missing gradient on prunable weight"
        assert layer.gate_scores.grad is not None, "Missing gradient on gate_scores"

    dummy_images = torch.randn(16, 3, 32, 32)
    dummy_labels = torch.randint(0, 10, (16,))
    dummy_loader = DataLoader(TensorDataset(dummy_images, dummy_labels), batch_size=8)
    acc = evaluate(model, dummy_loader, device=device, temperature=1.0)
    assert 0.0 <= acc <= 100.0, "Accuracy should be a percentage in [0, 100]"

    sparsity = compute_sparsity_level(model, threshold=1e-2, temperature=1.0)
    assert 0.0 <= sparsity <= 100.0, "Sparsity should be a percentage in [0, 100]"


def _part4_hard_pruning_sanity_check(device: torch.device) -> None:
    """Ensure hard pruning creates exact zeros for pruned connections."""
    model = SelfPruningNet().to(device)
    prunable_layers = _get_prunable_layers(model)

    with torch.no_grad():
        for layer in prunable_layers:
            layer.weight.fill_(1.0)
            layer.gate_scores.fill_(0.0)

        first_layer = prunable_layers[0]
        first_layer.gate_scores[0, 0] = -20.0
        first_layer.gate_scores[0, 1] = 20.0

    threshold = 1e-2
    hard_model = apply_hard_pruning(model, threshold=threshold, temperature=1.0)
    hard_first_layer = _get_prunable_layers(hard_model)[0]

    assert hard_first_layer.weight[0, 0].item() == 0.0, "Pruned weight should be exactly zero"
    assert hard_first_layer.weight[0, 1].item() != 0.0, "Active weight should be preserved"
    assert hard_first_layer.gate_scores[0, 0].item() < -1e5, "Pruned gate score should be strongly negative"


def _part5_compression_metrics_sanity_check(device: torch.device) -> None:
    """Validate compression metric calculations on a controlled setup."""
    model = nn.Sequential(PrunableLinear(2, 2)).to(device)
    layer = model[0]

    with torch.no_grad():
        layer.gate_scores.copy_(
            torch.tensor(
                [
                    [-20.0, 20.0],
                    [0.0, -20.0],
                ],
                device=device,
            )
        )

    metrics = compute_compression_metrics(model, threshold=1e-2, temperature=1.0)

    assert metrics["total_weights"] == 4.0, "Total weights mismatch"
    assert metrics["active_weights"] == 2.0, "Active weights mismatch"
    assert metrics["pruned_weights"] == 2.0, "Pruned weights mismatch"
    assert abs(metrics["sparsity_pct"] - 50.0) < 1e-6, "Sparsity percentage mismatch"
    assert abs(metrics["compression_ratio"] - 2.0) < 1e-6, "Compression ratio mismatch"
    assert metrics["total_flops"] == 8.0, "Total FLOPs mismatch"
    assert metrics["active_flops"] == 4.0, "Active FLOPs mismatch"
    assert abs(metrics["flops_reduction_pct"] - 50.0) < 1e-6, "FLOPs reduction mismatch"


def _part6_visualization_sanity_check() -> None:
    """Generate all Part 6 plots from synthetic results and verify files."""
    lambda_values = [1e-5, 1e-4, 1e-3]
    dummy_results = _build_dummy_results_for_reporting(lambda_values)

    with tempfile.TemporaryDirectory(prefix="part6_plots_") as tmp_dir:
        plot_paths = generate_part6_plots(
            results=dummy_results,
            lambda_values=lambda_values,
            output_dir=tmp_dir,
        )

        required_keys = [
            "gate_distribution",
            "gate_evolution",
            "sparsity_accuracy_tradeoff",
            "training_curves",
        ]
        for key in required_keys:
            path = Path(plot_paths[key])
            assert path.exists(), f"Missing plot file: {path}"


def _part7_report_sanity_check() -> None:
    """Generate report from synthetic results and verify key sections."""
    lambda_values = [1e-5, 1e-4, 1e-3]
    dummy_results = _build_dummy_results_for_reporting(lambda_values)

    with tempfile.TemporaryDirectory(prefix="part7_report_") as tmp_dir:
        plot_paths = generate_part6_plots(
            results=dummy_results,
            lambda_values=lambda_values,
            output_dir=tmp_dir,
        )
        report_path = generate_part7_report(
            results=dummy_results,
            lambda_values=lambda_values,
            plot_paths=plot_paths,
            output_dir=tmp_dir,
            report_filename="report.md",
        )

        report_file = Path(report_path)
        assert report_file.exists(), "Missing report file"
        report_text = report_file.read_text(encoding="utf-8")
        assert "# Self-Pruning Neural Network - Report" in report_text, "Missing report title"
        assert "## 2. Results" in report_text, "Missing results section"
        assert "### Best vs Final Epoch Summary" in report_text, "Missing best-vs-final table"
        assert "## 3. Gate Distribution" in report_text, "Missing plot section"


def _parse_lambdas(raw: str) -> List[float]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    if not values:
        raise ValueError("At least one lambda value must be provided.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-pruning network for CIFAR-10")
    parser.add_argument("--mode", choices=["sanity", "train"], default="sanity")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pruning-threshold", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--gate-lr-multiplier", type=float, default=10.0)
    parser.add_argument("--lambdas", type=str, default="1e-4,5e-4,1e-3")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--report-filename", type=str, default="report.md")
    args, unknown = parser.parse_known_args()

    if unknown:
        print(f"Ignoring unrecognized arguments: {unknown}")

    return args


if __name__ == "__main__":
    args = parse_args()
    set_reproducibility(args.seed)

    runtime_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        pruning_threshold=args.pruning_threshold,
        grad_clip_max_norm=args.grad_clip,
        gate_lr_multiplier=args.gate_lr_multiplier,
        lambda_values=_parse_lambdas(args.lambdas),
    )

    if args.mode == "sanity":
        _part1_gradient_flow_sanity_check(runtime_device)
        _part2_sparsity_loss_sanity_check(runtime_device)
        _part3_pipeline_sanity_check(runtime_device)
        _part4_hard_pruning_sanity_check(runtime_device)
        _part5_compression_metrics_sanity_check(runtime_device)
        _part6_visualization_sanity_check()
        _part7_report_sanity_check()
        print("Part 1 complete: PrunableLinear implemented and gradient check passed.")
        print("Part 2 complete: L1 gate-sum sparsity loss implemented and sanity checks passed.")
        print("Part 3 complete: model/data/training pipeline components implemented.")
        print("Part 4 complete: hard pruning implemented and sanity check passed.")
        print("Part 5 complete: compression/FLOPs metrics implemented and sanity check passed.")
        print("Part 6 complete: all visualization plots implemented and sanity check passed.")
        print("Part 7 complete: auto-generated report implemented and sanity check passed.")
    else:
        train_loader, test_loader = get_cifar10_loaders(config=config, data_dir=args.data_dir)
        all_results = run_part3_experiments(
            train_loader=train_loader,
            test_loader=test_loader,
            device=runtime_device,
            config=config,
        )
        assert_sparsity_monotonicity(all_results, config.lambda_values)
        print("Best-epoch sparsity monotonicity check passed for provided lambda values.")
        plots_output_dir = Path(args.output_dir) / "Plots"
        plot_paths = generate_part6_plots(
            results=all_results,
            lambda_values=config.lambda_values,
            output_dir=str(plots_output_dir),
        )
        report_path = generate_part7_report(
            results=all_results,
            lambda_values=config.lambda_values,
            plot_paths=plot_paths,
            output_dir=args.output_dir,
            report_filename=args.report_filename,
        )

        print("\nPart 3 experiments finished.")
        for lambda_value in config.lambda_values:
            run_result = all_results[lambda_value]
            hard_metrics = run_result["compression_metrics_hard"]
            print(
                f"lambda={lambda_value:.1e} | "
                f"best_epoch={run_result['best_epoch']} | "
                f"best_test_acc={run_result['best_test_accuracy']:.2f}% | "
                f"final_test_acc={run_result['final_test_accuracy']:.2f}% | "
                f"hard_pruned_test_acc={run_result['hard_pruned_test_accuracy']:.2f}% | "
                f"best_sparsity={run_result['best_sparsity_percent']:.2f}% | "
                f"final_sparsity={run_result['final_sparsity_percent']:.2f}% | "
                f"compression={hard_metrics['compression_ratio']:.2f}x | "
                f"flops_reduction={hard_metrics['flops_reduction_pct']:.2f}%"
            )

        print("\nPart 6 plots saved:")
        print(f"best_lambda={plot_paths['best_lambda']}")
        print(f"gate_distribution={plot_paths['gate_distribution']}")
        print(f"gate_evolution={plot_paths['gate_evolution']}")
        print(f"sparsity_accuracy_tradeoff={plot_paths['sparsity_accuracy_tradeoff']}")
        print(f"training_curves={plot_paths['training_curves']}")
        print(f"report={report_path}")
