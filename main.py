"""
CLI Entry Point for SNN Training

Usage:
    python -m refactored_snn_training_testing.main [options]
    python main.py [options]
"""

import argparse
import os
import sys

# Handle both direct execution and module execution
if __name__ == "__main__" and __package__ is None:
    # Running as script - add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from refactored_snn_training_testing.config import SNNConfig
    from refactored_snn_training_testing.train import train
else:
    # Running as module
    from .config import SNNConfig
    from .train import train


def main():
    parser = argparse.ArgumentParser(
        description="Train 3-layer SNN on Primate Reaching dataset"
    )

    # Data options
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory containing dataset files",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="indy_20160622_01.mat",
        help="Dataset filename",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download dataset if not found",
    )

    # Training options
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")

    # Model options
    parser.add_argument("--hidden-size", type=int, default=50, help="Hidden layer size")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--window", type=int, default=50, help="Temporal window size")

    # Other options
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-dir", type=str, default=".", help="Directory to save model")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification mode (first batch only)",
    )

    args = parser.parse_args()

    # Create config
    config = SNNConfig(
        data_dir=args.data_dir,
        filename=args.filename,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        hidden_size=args.hidden_size,
        window=args.window,
        seed=args.seed,
    )

    # Train
    result = train(
        config,
        save_dir=args.save_dir,
        verify_mode=args.verify,
        download=args.download,
    )

    if args.verify:
        print("\nVerification mode results:")
        print(f"  Input shape: {result['x'].shape}")
        print(f"  Output shape: {result['predictions'].shape}")
        print(f"  Loss: {result['loss']:.6f}")
        print(f"  fc1 gradient shape: {result['fc1_grad'].shape}")


if __name__ == "__main__":
    main()
