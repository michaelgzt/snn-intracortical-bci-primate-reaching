"""
Testing and Benchmarking for SNN

Inference utilities and comprehensive benchmarking.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import SNNConfig
from .dataset import load_primate_reaching_data, PrimateReachingDataset
from .model import SNN3
from .metrics import (
    compute_footprint,
    compute_connection_sparsity,
    compute_r2,
    compute_mse,
    compute_activation_sparsity,
    compute_synaptic_operations,
)


def run_inference(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple:
    """
    Run inference on test set.

    Returns:
        predictions: All predictions (N, 2)
        labels: All labels (N, 2)
    """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            predictions = model(x)

            # Take last timestep
            pred_last = predictions[:, :, -1]
            y_last = y[:, :, -1]

            all_predictions.append(pred_last.cpu())
            all_labels.append(y_last.cpu())

    return torch.cat(all_predictions), torch.cat(all_labels)


def run_inference_with_captures(
    model: nn.Module,
    inputs: torch.Tensor,
) -> dict:
    """
    Run forward pass while capturing intermediate activations.

    Args:
        model: SNN3 model instance (must be in eval mode)
        inputs: Input tensor, shape (timesteps, 1, input_size)

    Returns:
        Dictionary with predictions and captured activations
    """
    model.reset()

    predictions = []
    all_spk1, all_spk2, all_spk3 = [], [], []
    all_inp_fc1, all_inp_fc2, all_inp_fc3, all_inp_fc_out = [], [], [], []

    for t in range(inputs.shape[0]):
        x = inputs[t].squeeze()

        # Layer 1
        all_inp_fc1.append(x.clone())
        cur1 = model.fc1(x)
        spk1, model.mem1 = model.lif1(cur1, model.mem1)
        all_spk1.append(spk1.clone())

        # Layer 2
        all_inp_fc2.append(spk1.clone())
        cur2 = model.fc2(spk1)
        spk2, model.mem2 = model.lif2(cur2, model.mem2)
        all_spk2.append(spk2.clone())

        # Layer 3
        all_inp_fc3.append(spk2.clone())
        cur3 = model.fc3(spk2)
        spk3, model.mem3 = model.lif3(cur3, model.mem3)
        all_spk3.append(spk3.clone())

        # Output
        all_inp_fc_out.append(spk3.clone())
        cur_out = model.fc_out(spk3)
        _, model.mem_out = model.lif_out(cur_out, model.mem_out)
        predictions.append(model.mem_out.clone())

    return {
        "predictions": torch.stack(predictions),
        "all_spk1": all_spk1,
        "all_spk2": all_spk2,
        "all_spk3": all_spk3,
        "all_inp_fc1": all_inp_fc1,
        "all_inp_fc2": all_inp_fc2,
        "all_inp_fc3": all_inp_fc3,
        "all_inp_fc_out": all_inp_fc_out,
    }


def benchmark_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Run complete benchmark with all metrics.

    Returns:
        Dictionary with all benchmark metrics
    """
    model.eval()

    # Static metrics
    footprint = compute_footprint(model)
    connection_sparsity = compute_connection_sparsity(model)

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.squeeze()

            inputs = x

            # Run inference and capture intermediate values
            results = run_inference_with_captures(model, inputs)
        r2 = compute_r2(results['predictions'].cpu(), y)
        mse = compute_mse(results['predictions'].cpu(), y)
        activation_sparsity = compute_activation_sparsity(results['all_spk1'], results['all_spk2'], results['all_spk3'])

        synaptic_ops = compute_synaptic_operations(
            model,
            {
                "all_inp_fc1": results['all_inp_fc1'],
                "all_inp_fc2": results['all_inp_fc2'],
                "all_inp_fc3": results['all_inp_fc3'],
                "all_inp_fc_out": results['all_inp_fc_out'],
            },
            y.shape[0],
        )

    return {
        "r2": r2,
        "mse": mse,
        "footprint": footprint,
        "connection_sparsity": connection_sparsity,
        "activation_sparsity": activation_sparsity,
        "synaptic_operations": synaptic_ops,
    }


def print_benchmark_results(results: dict) -> None:
    """Print benchmark results in formatted output."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  R² Score:             {results['r2']:.6f}")
    print(f"  MSE:                  {results['mse']:.6f}")
    print(f"  Footprint:            {results['footprint']:,} bytes")
    print(f"  Connection Sparsity:  {results['connection_sparsity']:.4f}")
    print(f"  Activation Sparsity:  {results['activation_sparsity']:.6f}")
    print("\n  Synaptic Operations:")
    print(f"    Effective MACs:     {results['synaptic_operations']['Effective_MACs']:.2f}")
    print(f"    Effective ACs:      {results['synaptic_operations']['Effective_ACs']:.2f}")
    print(f"    Dense:              {results['synaptic_operations']['Dense']:.2f}")
    print("=" * 60)


def load_and_benchmark(
    model_path: str,
    config: SNNConfig,
    device: torch.device = None,
    download: bool = True,
) -> dict:
    """
    Load a trained model and run benchmark.

    Args:
        model_path: Path to model checkpoint
        config: Configuration object
        device: Device to use
        download: Whether to download dataset if not found

    Returns:
        Benchmark results dictionary
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load data
    data = load_primate_reaching_data(
        data_dir=config.data_dir,
        filename=config.filename,
        bin_width=config.bin_width,
        stride=config.stride,
        download=download,
    )

    # Create test dataset
    test_dataset = PrimateReachingDataset(
        data["samples"], data["labels"], data["ind_test"],
        1, data["ratio"]
    )
    test_loader = DataLoader(test_dataset, batch_size=len(data["ind_test"]), shuffle=False)

    # Load model
    model = SNN3(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        output_size=config.output_size,
        tau=config.tau,
        threshold=config.threshold,
        dropout_p=config.dropout,
        device=str(device),
    )

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")

    # Run benchmark
    results = benchmark_model(model, test_loader, device)

    return results
