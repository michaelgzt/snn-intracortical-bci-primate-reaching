"""
Training Loop for SNN

Training with AdamW optimizer and early stopping.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import SNNConfig
from .dataset import load_primate_reaching_data, PrimateReachingDataset
from .model import SNN3
from .loss import temporal_weighted_mse
from .metrics import (
    compute_footprint,
    compute_connection_sparsity,
)
from .test import benchmark_model


def get_device() -> torch.device:
    """Get best available device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    window: int,
    device: torch.device,
) -> float:
    """
    Train for one epoch.

    Returns:
        Average loss over all batches
    """
    model.train()
    total_loss = 0
    total_samples = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        predictions = model(x)

        # Compute loss
        loss = temporal_weighted_mse(predictions, y, window, device)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.shape[0]
        total_samples += x.shape[0]

    return total_loss / total_samples


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Validate model on validation set.

    Returns:
        Average MSE on the last timestep prediction
    """
    model.eval()
    total_loss = 0
    total_samples = 0
    mse_fn = nn.MSELoss()

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            predictions = model(x)

            # Evaluate on last timestep only
            pred_last = predictions[:, :, -1]
            y_last = y[:, :, -1]

            loss = mse_fn(pred_last, y_last)

            total_loss += loss.item() * x.shape[0]
            total_samples += x.shape[0]

    return total_loss / total_samples




def save_checkpoint(
    model: nn.Module,
    epoch: int,
    metrics: dict,
    path: str,
) -> None:
    """Save model checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "test_metrics": metrics,
    }, path)


def load_checkpoint(
    path: str,
    model: nn.Module = None,
    device: torch.device = None,
) -> dict:
    """
    Load model checkpoint.

    Args:
        path: Path to checkpoint file
        model: Optional model to load state into
        device: Device to load onto

    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def train(
    config: SNNConfig,
    save_dir: str = None,
    verify_mode: bool = False,
    download: bool = True,
):
    """
    Main training function.

    Args:
        config: Configuration object
        save_dir: Directory to save model (default: current directory)
        verify_mode: If True, only run first batch and return debug info
        download: Whether to download dataset if not found

    Returns:
        Trained model (or debug info in verify_mode)
    """
    # Set seeds
    set_seed(config.seed)

    # Device
    device = get_device() if config.device is None else torch.device(config.device)
    print(f"Using device: {device}")

    # Load data
    data = load_primate_reaching_data(
        data_dir=config.data_dir,
        filename=config.filename,
        bin_width=config.bin_width,
        stride=config.stride,
        download=download,
    )

    # Create datasets
    train_dataset = PrimateReachingDataset(
        data["samples"], data["labels"], data["ind_train"],
        config.window, data["ratio"]
    )
    val_dataset = PrimateReachingDataset(
        data["samples"], data["labels"], data["ind_val"],
        config.window, data["ratio"]
    )
    test_dataset = PrimateReachingDataset(
        data["samples"], data["labels"], data["ind_test"],
        1, data["ratio"]
    )

    # Create data loaders (reset seed for same shuffling)
    set_seed(config.seed)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    # Test loader uses batch_size=1 for step-by-step evaluation (same as test.py)
    test_loader = DataLoader(test_dataset, batch_size=len(data["ind_test"]), shuffle=False)

    # Create model (reset seed for same initialization)
    set_seed(config.seed)
    model = SNN3(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        output_size=config.output_size,
        tau=config.tau,
        threshold=config.threshold,
        dropout_p=config.dropout,
        device=str(device),
    )
    model.to(device)

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # Verification mode
    if verify_mode:
        model.train()
        x, y = next(iter(train_loader))
        x = x.to(device)
        y = y.to(device)

        predictions = model(x)
        loss = temporal_weighted_mse(predictions, y, config.window, device)

        optimizer.zero_grad()
        loss.backward()

        return {
            "x": x.cpu(),
            "y": y.cpu(),
            "predictions": predictions.cpu().detach(),
            "loss": loss.item(),
            "fc1_grad": model.fc1.weight.grad.cpu().clone(),
            "fc1_weight": model.fc1.weight.cpu().detach().clone(),
        }

    # Print static metrics
    print(f"\nModel Footprint: {compute_footprint(model):,} bytes")
    print(f"Initial Connection Sparsity: {compute_connection_sparsity(model):.2%}")

    # Evaluate initial state before training
    print("\n" + "=" * 80)
    print("Initial state (before training)...")
    print("=" * 80)
    initial_val_loss = validate(model, val_loader, device)
    initial_test_metrics = benchmark_model(model, test_loader, device)
    print(f"Initial Val Loss = {initial_val_loss:.6f} | "
          f"Test: MSE={initial_test_metrics['mse']:.6f}, "
          f"R²={initial_test_metrics['r2']:.4f}, "
          f"Sparsity={initial_test_metrics['activation_sparsity']:.2%}")

    # Training loop
    best_val_loss = initial_val_loss
    best_epoch = -1  # -1 indicates initial state (before training)
    # Include data session in model filename (e.g., "best_model_indy_20160622_01.pt")
    session_name = config.filename.replace(".mat", "")
    save_path = os.path.join(save_dir or ".", f"best_model_{session_name}.pt")

    # Save initial checkpoint (in case training doesn't improve)
    save_checkpoint(model, -1, initial_test_metrics, save_path)

    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, config.window, device)
        val_loss = validate(model, val_loader, device)
        test_metrics = benchmark_model(model, test_loader, device)

        print(f"Epoch {epoch:3d}: "
              f"Train Loss = {train_loss:.6f}, "
              f"Val Loss = {val_loss:.6f} | "
              f"Test: MSE={test_metrics['mse']:.6f}, "
              f"R²={test_metrics['r2']:.4f}, "
              f"Sparsity={test_metrics['activation_sparsity']:.2%}")

        # Early stopping
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(model, epoch, test_metrics, save_path)
        elif epoch >= 20 and (epoch - best_epoch) >= config.patience:
            print(f"\nEarly stopping at epoch {epoch} (best was {best_epoch})")
            break

    # Load best model
    print("\n" + "=" * 80)
    print("Training complete. Loading best model...")
    print("=" * 80)

    checkpoint = load_checkpoint(save_path, model, device)
    if checkpoint['epoch'] == -1:
        print("Loaded initial model (no improvement during training)")
    else:
        print(f"Loaded best model from epoch {checkpoint['epoch']}")

    # Final test
    final_metrics = benchmark_model(model, test_loader, device)

    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)
    print(f"  MSE:                  {final_metrics['mse']:.6f}")
    print(f"  R² Score:             {final_metrics['r2']:.4f}")
    print(f"  Activation Sparsity:  {final_metrics['activation_sparsity']:.2%}")
    print(f"  Connection Sparsity:  {final_metrics['connection_sparsity']:.4f}")
    print(f"  Model Footprint:      {final_metrics['footprint']:,} bytes")
    print(f"  Effective ACs:        {final_metrics['synaptic_operations']['Effective_ACs']:.2f}")
    print("=" * 80)

    return model
