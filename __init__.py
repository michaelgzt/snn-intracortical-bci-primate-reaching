"""
Refactored SNN Training and Testing

A modular codebase for training and testing Spiking Neural Networks
on the Primate Reaching dataset.
"""

from config import SNNConfig
from model import SNN3
from dataset import load_primate_reaching_data, PrimateReachingDataset
from metrics import (
    compute_footprint,
    compute_connection_sparsity,
    compute_r2,
    compute_pearson,
    compute_mse,
    compute_activation_sparsity,
    compute_synaptic_operations,
)
from loss import temporal_weighted_mse
from train import train
from test import benchmark_model

__all__ = [
    "SNNConfig",
    "SNN3",
    "load_primate_reaching_data",
    "PrimateReachingDataset",
    "compute_footprint",
    "compute_connection_sparsity",
    "compute_r2",
    "compute_pearson",
    "compute_mse",
    "compute_activation_sparsity",
    "compute_synaptic_operations",
    "temporal_weighted_mse",
    "train",
    "benchmark_model",
]
