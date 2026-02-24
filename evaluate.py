"""
CLI Entry Point for SNN Evaluation

Usage:
    python -m refactored_snn_training_testing.evaluate --model path/to/model.pt
    python evaluate.py --model path/to/model.pt
"""

import argparse
import os
import sys

# Handle both direct execution and module execution
if __name__ == "__main__" and __package__ is None:
    # Running as script - add current directory to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import SNNConfig
    from test import load_and_benchmark, print_benchmark_results
else:
    # Running as module
    from config import SNNConfig
    from test import load_and_benchmark, print_benchmark_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained SNN on Primate Reaching dataset"
    )

    # Model options
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint",
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

    # Model options (must match training)
    parser.add_argument("--hidden-size", type=int, default=50, help="Hidden layer size")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--window", type=int, default=50, help="Temporal window size")

    args = parser.parse_args()

    # Create config
    config = SNNConfig(
        data_dir=args.data_dir,
        filename=args.filename,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        window=args.window,
    )

    # Run benchmark
    print(f"Loading model from: {args.model}")
    results = load_and_benchmark(
        args.model,
        config,
        download=args.download,
    )

    # Print results
    print_benchmark_results(results)


if __name__ == "__main__":
    main()
