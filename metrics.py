"""
Evaluation Metrics for SNN

Static metrics (computed from model):
    - Footprint: Total memory size in bytes
    - Connection Sparsity: Fraction of zero weights

Workload metrics (computed during inference):
    - R2 Score: Coefficient of determination
    - Pearson Correlation: Linear correlation coefficient
    - MSE: Mean squared error
    - Activation Sparsity: Fraction of zero activations
    - Synaptic Operations: Count of MAC/AC operations
"""

import torch
import torch.nn as nn


# =============================================================================
# STATIC METRICS
# =============================================================================

def compute_footprint(model: nn.Module) -> int:
    """
    Compute model memory footprint in bytes.

    Counts memory for all parameters and buffers.
    """
    param_size = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )
    buffer_size = sum(
        b.numel() * b.element_size()
        for b in model.buffers()
    )
    return param_size + buffer_size


def compute_connection_sparsity(model: nn.Module) -> float:
    """
    Compute fraction of zero weights in Linear layers.

    Dense networks have sparsity = 0.0.
    Fully pruned networks have sparsity = 1.0.
    """
    count_zeros = 0
    count_total = 0

    for module in model.modules():
        if isinstance(module, nn.Linear):
            count_zeros += (module.weight == 0).sum().item()
            count_total += module.weight.numel()

    if count_total == 0:
        return 0.0

    return count_zeros / count_total


# =============================================================================
# WORKLOAD METRICS
# =============================================================================

def compute_r2(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute R-squared for 2D velocity prediction.

    R² = 1 - SS_residual / SS_total

    Args:
        predictions: shape (N, 2) or (N, 2, T)
        labels: shape (N, 2) or (N, 2, T)

    Returns:
        Average R² across x and y dimensions
    """
    # Flatten if needed
    if predictions.dim() == 3:
        predictions = predictions[:, :, -1]  # Take last timestep
        labels = labels[:, :, -1]

    # X dimension
    x_true = labels[:, 0]
    x_pred = predictions[:, 0]
    x_ss_res = torch.sum((x_true - x_pred) ** 2).item()
    x_ss_tot = x_true.var(correction=0).item() * len(x_true)
    x_r2 = 1 - (x_ss_res / x_ss_tot) if x_ss_tot > 0 else 0.0

    # Y dimension
    y_true = labels[:, 1]
    y_pred = predictions[:, 1]
    y_ss_res = torch.sum((y_true - y_pred) ** 2).item()
    y_ss_tot = y_true.var(correction=0).item() * len(y_true)
    y_r2 = 1 - (y_ss_res / y_ss_tot) if y_ss_tot > 0 else 0.0

    return (x_r2 + y_r2) / 2


def compute_pearson(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute Pearson correlation coefficient for 2D velocity prediction.

    Args:
        predictions: shape (N, 2) or (N, 2, T)
        labels: shape (N, 2) or (N, 2, T)

    Returns:
        Average Pearson r across x and y dimensions
    """
    # Flatten if needed
    if predictions.dim() == 3:
        predictions = predictions[:, :, -1]
        labels = labels[:, :, -1]

    # X dimension
    x_pred = predictions[:, 0]
    x_true = labels[:, 0]
    x_pred_centered = x_pred - x_pred.mean()
    x_true_centered = x_true - x_true.mean()
    x_cov = (x_pred_centered * x_true_centered).mean()
    x_std = x_pred.std() * x_true.std()
    x_r = (x_cov / x_std).item() if x_std > 0 else 0.0

    # Y dimension
    y_pred = predictions[:, 1]
    y_true = labels[:, 1]
    y_pred_centered = y_pred - y_pred.mean()
    y_true_centered = y_true - y_true.mean()
    y_cov = (y_pred_centered * y_true_centered).mean()
    y_std = y_pred.std() * y_true.std()
    y_r = (y_cov / y_std).item() if y_std > 0 else 0.0

    return (x_r + y_r) / 2


def compute_mse(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute MSE for 2D velocity prediction.

    Args:
        predictions: shape (N, 2) or (N, 2, T)
        labels: shape (N, 2) or (N, 2, T)

    Returns:
        Average MSE across x and y dimensions
    """
    if predictions.dim() == 3:
        predictions = predictions[:, :, -1]
        labels = labels[:, :, -1]

    return nn.functional.mse_loss(predictions, labels).item()


def compute_activation_sparsity(
    all_spk1: list,
    all_spk2: list,
    all_spk3: list = None,
) -> float:
    """
    Compute activation sparsity from captured spike outputs.

    Sparsity = (total_activations - nonzero_activations) / total_activations

    Args:
        all_spk1: List of spike tensors from layer 1
        all_spk2: List of spike tensors from layer 2
        all_spk3: Optional list of spike tensors from layer 3 (for 3-layer models)

    Returns:
        Fraction of zero activations (higher = more efficient)
    """
    spk1 = torch.stack(all_spk1)
    spk2 = torch.stack(all_spk2)

    total = spk1.numel() + spk2.numel()
    nonzero = torch.count_nonzero(spk1).item() + torch.count_nonzero(spk2).item()

    if all_spk3 is not None:
        spk3 = torch.stack(all_spk3)
        total += spk3.numel()
        nonzero += torch.count_nonzero(spk3).item()

    return (total - nonzero) / total


def compute_synaptic_operations(
    model: nn.Module,
    all_inputs: dict,
    num_samples: int,
) -> dict:
    """
    Compute synaptic operations (MACs/ACs) for the network.

    For SNNs (binary spikes):
        - Effective_MACs: 0 (no multiplies, just accumulates)
        - Effective_ACs: Actual accumulate operations
        - Dense: Theoretical maximum if nothing sparse

    Args:
        model: SNN model with fc1, fc2, fc3, fc_out layers
        all_inputs: Dict with keys 'all_inp_fc1', 'all_inp_fc2', 'all_inp_fc3', 'all_inp_fc_out'
        num_samples: Number of samples for normalization

    Returns:
        Dict with Effective_MACs, Effective_ACs, Dense
    """
    total_acs = 0
    total_dense = 0

    layers_and_inputs = [
        (model.fc1, all_inputs.get("all_inp_fc1", [])),
        (model.fc2, all_inputs.get("all_inp_fc2", [])),
        (model.fc3, all_inputs.get("all_inp_fc3", [])),
        (model.fc_out, all_inputs.get("all_inp_fc_out", [])),
    ]

    for layer, inputs_list in layers_and_inputs:
        if not inputs_list:
            continue

        for inp in inputs_list:
            inp_bin = inp.clone().detach()
            weight_bin = layer.weight.data.clone().detach()

            # Binarize: non-zero values become 1
            inp_bin[inp_bin != 0] = 1
            weight_bin[weight_bin != 0] = 1

            # Effective operations: count where both weight and input are non-zero
            effective_ops = (weight_bin @ inp_bin).sum().item()

            # Dense operations: what if every connection was active?
            inp_ones = torch.ones_like(inp_bin)
            weight_ones = torch.ones_like(weight_bin)
            dense_ops = (weight_ones @ inp_ones).sum().item()

            total_acs += effective_ops
            total_dense += dense_ops

    return {
        "Effective_MACs": 0.0,
        "Effective_ACs": total_acs / num_samples if num_samples > 0 else 0.0,
        "Dense": total_dense / num_samples if num_samples > 0 else 0.0,
    }
