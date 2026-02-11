"""
Configuration for SNN Training and Testing

All hyperparameters are centralized here with default values
matching the original training code.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SNNConfig:
    """Configuration dataclass for SNN training and testing."""

    # Data paths
    data_dir: str = "./data"
    filename: str = "indy_20160622_01.mat"

    # Training hyperparameters
    batch_size: int = 256
    epochs: int = 100
    lr: float = 0.001
    weight_decay: float = 0.01
    dropout: float = 0.2
    patience: int = 20

    # SNN architecture
    input_size: int = 96        # 96 for indy files, 192 for loco files
    hidden_size: int = 50       # neurons per hidden layer
    output_size: int = 2        # x, y velocity

    # LIF neuron parameters
    tau: float = 0.96           # membrane decay factor (beta)
    threshold: float = 1.0      # spike threshold voltage

    # Temporal parameters
    window: int = 50            # time steps to unfold SNN
    warmup_steps: int = 10      # steps to discard in loss
    stride: float = 0.004       # stride in seconds (4ms)
    bin_width: float = 0.004    # spike binning width in seconds (4ms)

    # Reproducibility
    seed: int = 42

    # Device (auto-detected if None)
    device: Optional[str] = None

    def __post_init__(self):
        """Validate configuration and auto-detect input size from filename."""
        if "indy" in self.filename:
            self.input_size = 96
        elif "loco" in self.filename:
            self.input_size = 192

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "data_dir": self.data_dir,
            "filename": self.filename,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "patience": self.patience,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "tau": self.tau,
            "threshold": self.threshold,
            "window": self.window,
            "warmup_steps": self.warmup_steps,
            "stride": self.stride,
            "bin_width": self.bin_width,
            "seed": self.seed,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SNNConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
