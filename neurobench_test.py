"""
NeuroBench Benchmark Test for SNN3 Model

This script benchmarks the trained SNN3 model using the NeuroBench framework,
following the pattern from the Primate Reaching tutorial.
"""

import os
import torch
import torch.nn as nn
import snntorch as snn
from torch.utils.data import DataLoader, Subset

from neurobench.datasets import PrimateReaching
from neurobench.models import TorchModel
from neurobench.benchmarks import Benchmark
from neurobench.metrics.workload import (
    ActivationSparsity,
    SynapticOperations,
    R2
)
from neurobench.metrics.static import (
    Footprint,
    ConnectionSparsity,
)

from model import SNN3


class SNN3NeuroBenchWrapper(nn.Module):
    """Wrapper to make SNN3 compatible with NeuroBench streaming format.

    Processes one timestep at a time, maintaining membrane state between
    timesteps (exactly like SNN2 does in the tutorial).

    Input: (len_series, 1, features) - entire test set as one batch
    Output: (len_series, 2) - velocity predictions
    """

    def __init__(self, snn3_model):
        super().__init__()
        self.snn3 = snn3_model
        # Membrane potentials maintained between timesteps
        self.mem1 = None
        self.mem2 = None
        self.mem3 = None
        self.mem_out = None

    def reset(self):
        """Reset membrane potentials to initial state."""
        self.mem1 = self.snn3.lif1.init_leaky()
        self.mem2 = self.snn3.lif2.init_leaky()
        self.mem3 = self.snn3.lif3.init_leaky()
        self.mem_out = self.snn3.lif_out.init_leaky()

    def single_forward(self, x):
        """Process single timestep.

        Args:
            x: Input tensor of shape (1, features) or (features,)

        Returns:
            Membrane potential output of shape (2,)
        """
        x = x.squeeze()  # (features,)

        # Layer 1
        cur1 = self.snn3.fc1(x)
        spk1, self.mem1 = self.snn3.lif1(cur1, self.mem1)

        # Layer 2
        cur2 = self.snn3.fc2(spk1)
        spk2, self.mem2 = self.snn3.lif2(cur2, self.mem2)

        # Layer 3
        cur3 = self.snn3.fc3(spk2)
        spk3, self.mem3 = self.snn3.lif3(cur3, self.mem3)

        # Output layer
        cur_out = self.snn3.fc_out(spk3)
        _, self.mem_out = self.snn3.lif_out(cur_out, self.mem_out)

        return self.mem_out.clone()  # (2,)

    def forward(self, x):
        """Forward pass over entire time series.

        Args:
            x: Input tensor of shape (len_series, 1, features)

        Returns:
            Predictions of shape (len_series, 2)
        """
        predictions = []
        for t in range(x.shape[0]):
            pred = self.single_forward(x[t, ...])
            predictions.append(pred)
        return torch.stack(predictions)


def run_neurobench_benchmark(
    model_path: str,
    data_dir: str,
    filename: str,
    device: torch.device = None,
) -> dict:
    """
    Run NeuroBench benchmark on a trained SNN3 model.

    Args:
        model_path: Path to the trained model checkpoint
        data_dir: Directory containing the dataset
        filename: Dataset filename (e.g., 'indy_20160622_01')
        device: Torch device to use

    Returns:
        Dictionary of benchmark results
    """
    if device is None:
        device = torch.device("cpu")

    # Load NeuroBench dataset (same settings as tutorial)
    dataset = PrimateReaching(
        file_path=data_dir,
        filename=filename,
        num_steps=1,  # Single timestep per sample
        train_ratio=0.5,
        bin_width=0.004,  # 4ms bins
        biological_delay=0,
        remove_segments_inactive=False,
    )

    # Load ENTIRE test set as single batch (like SNN2 in tutorial)
    test_set_loader = DataLoader(
        Subset(dataset, dataset.ind_test),
        batch_size=len(dataset.ind_test),
        shuffle=False,
    )

    # Determine input size from filename
    input_size = 96 if "indy" in filename else 192

    # Load trained SNN3 model
    net = SNN3(
        input_size=input_size,
        hidden_size=50,
        output_size=2,
        tau=0.96,
        threshold=1.0,
        dropout_p=0.2,
        device=str(device),
    )

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.to(device)
    net.eval()

    # Wrap for NeuroBench compatibility
    wrapped_net = SNN3NeuroBenchWrapper(net)
    wrapped_net.reset()
    wrapped_net.to(device)

    # Create NeuroBench model wrapper
    model = TorchModel(wrapped_net)
    model.add_activation_module(snn.SpikingNeuron)

    # Define metrics
    static_metrics = [Footprint, ConnectionSparsity]
    workload_metrics = [R2, ActivationSparsity, SynapticOperations]

    # Create and run benchmark
    benchmark = Benchmark(
        model,
        test_set_loader,
        [],  # No preprocessors
        [],  # No postprocessors
        [static_metrics, workload_metrics]
    )

    results = benchmark.run()

    return results


def print_results(results: dict, filename: str) -> None:
    """Print benchmark results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"NeuroBench Results for {filename}")
    print(f"{'='*60}")
    print(f"Static Metrics:")
    print(f"  Footprint:            {results['Footprint']:,} bytes")
    print(f"  Connection Sparsity:  {results['ConnectionSparsity']:.4f}")
    print(f"\nWorkload Metrics:")
    print(f"  R2 Score:             {results['R2']:.6f}")
    print(f"  Activation Sparsity:  {results['ActivationSparsity']:.6f}")
    print(f"\nSynaptic Operations:")
    print(f"  Effective MACs:       {results['SynapticOperations']['Effective_MACs']:.2f}")
    print(f"  Effective ACs:        {results['SynapticOperations']['Effective_ACs']:.2f}")
    print(f"  Dense:                {results['SynapticOperations']['Dense']:.2f}")
    print(f"{'='*60}")


def main():
    """Run NeuroBench benchmark on all trained models."""
    # Configuration
    file_path = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(file_path, "data")

    # Available trained models
    model_files = {
        "indy_20160622_01": "best_model_indy_20160622_01_42.pt",
        "indy_20160630_01": "best_model_indy_20160630_01_42.pt",
        "indy_20170131_02": "best_model_indy_20170131_02_42.pt",
        "loco_20170301_05": "best_model_loco_20170301_05_42.pt",
        "loco_20170215_02": "best_model_loco_20170215_02_42.pt",
        "loco_20170210_03": "best_model_loco_20170210_03_42.pt",
    }

    device = torch.device("cpu")

    # Collect results for summary
    all_results = {}

    for filename, model_file in model_files.items():
        print(f"\nProcessing {filename}...")

        model_path = os.path.join(file_path, model_file)

        if not os.path.exists(model_path):
            print(f"  Model not found: {model_path}")
            continue

        results = run_neurobench_benchmark(
            model_path=model_path,
            data_dir=data_dir,
            filename=filename,
            device=device,
        )

        all_results[filename] = results
        print_results(results, filename)

    # Print summary
    if all_results:
        print_summary(all_results)


def print_summary(all_results: dict) -> None:
    """Print summary of all benchmark results."""
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    # Extract metrics
    r2_scores = [r['R2'] for r in all_results.values()]
    act_sparsity = [r['ActivationSparsity'] for r in all_results.values()]
    acs = [r['SynapticOperations']['Effective_ACs'] for r in all_results.values()]

    print(f"\nR2 Scores:")
    for name, r in all_results.items():
        print(f"  {name}: {r['R2']:.6f}")
    print(f"  Average: {sum(r2_scores)/len(r2_scores):.6f}")

    print(f"\nActivation Sparsity:")
    for name, r in all_results.items():
        print(f"  {name}: {r['ActivationSparsity']:.6f}")
    print(f"  Average: {sum(act_sparsity)/len(act_sparsity):.6f}")

    print(f"\nEffective ACs:")
    for name, r in all_results.items():
        print(f"  {name}: {r['SynapticOperations']['Effective_ACs']:.2f}")
    print(f"  Average: {sum(acs)/len(acs):.2f}")


if __name__ == "__main__":
    main()
