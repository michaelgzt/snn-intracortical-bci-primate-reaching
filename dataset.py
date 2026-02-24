"""
Dataset Loading for Primate Reaching

Load and preprocess primate reaching spike data for SNN training and testing.
"""

import os
import math
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from scipy.signal import convolve2d

from download import ensure_dataset

# Sampling rate of spike data (4ms interval)
SAMPLING_RATE = 4e-3


def load_primate_reaching_data(
    data_dir: str,
    filename: str,
    bin_width: float = 0.004,
    stride: float = 0.004,
    train_ratio: float = 0.5,
    split_num: int = 1,
    download: bool = True,
) -> dict:
    """
    Load and preprocess primate reaching spike data.

    Data flow:
    1. Load raw spike times and cursor positions from HDF5 .mat file
    2. Convert spike times to binary spike trains via histogram binning
    3. Combine multiple units per channel via bitwise OR
    4. Apply temporal binning via convolution (if bin_width > SAMPLING_RATE)
    5. Convert cursor position to velocity (gradient)
    6. Split into train/val/test based on reaching segments

    Args:
        data_dir: Directory containing the dataset
        filename: Dataset filename (e.g., 'indy_20160622_01.mat')
        bin_width: Spike binning width in seconds (default 4ms)
        stride: Sampling stride in seconds (default 4ms)
        train_ratio: Fraction of data for training (default 0.5)
        split_num: Number of data chunks for splitting (default 4)
        download: Whether to download if not found (default True)

    Returns:
        Dictionary with:
            samples: tensor (channels, timesteps)
            labels: tensor (2, timesteps) - velocity
            ind_train: list of training indices
            ind_val: list of validation indices
            ind_test: list of test indices
            ratio: binning ratio
            input_feature_size: number of input channels
    """
    # Ensure dataset exists
    if download:
        file_path = ensure_dataset(data_dir, filename)
    else:
        file_path = os.path.join(data_dir, filename)

    print(f"Loading {filename}")

    # Load HDF5/MATLAB file
    dataset = h5py.File(file_path, "r")

    # Extract raw data
    spikes = dataset["spikes"][()]           # shape: (channels, units)
    cursor_pos = dataset["cursor_pos"][()]   # shape: (2, timesteps)
    target_pos = dataset["target_pos"][()]   # shape: (2, timesteps)
    t = np.squeeze(dataset["t"][()])         # timestamps

    # Create time bins aligned with spike recording interval
    new_t = np.arange(t[0] - bin_width, t[-1], SAMPLING_RATE)

    # ========================================
    # Convert spike times to binary spike trains
    # ========================================
    spike_train = np.zeros((*spikes.shape, len(new_t)), dtype=np.int8)

    for row_idx, row in enumerate(spikes):
        for col_idx, element in enumerate(row):
            # Handle both direct arrays and HDF5 references
            if isinstance(element, np.ndarray):
                spike_times = element
            else:
                spike_times = dataset[element][()]

            # Bin spikes using histogram
            bins, _ = np.histogram(spike_times, bins=new_t.squeeze())

            # Convert to binary: mark timesteps with spikes
            # +1 offset because histogram assigns to lower bound
            idx = np.nonzero(bins)[0] + 1
            spike_train[row_idx, col_idx, idx] = 1

    # ========================================
    # Combine units per channel via bitwise OR
    # ========================================
    spike_train = np.bitwise_or.reduce(spike_train, axis=0)

    # ========================================
    # Apply temporal binning via convolution
    # ========================================
    ratio = int(np.round(bin_width / SAMPLING_RATE))
    if ratio != 1:
        spike_train = convolve2d(spike_train, np.ones((1, ratio)), mode="valid")

    # ========================================
    # Create tensors
    # ========================================
    samples = torch.from_numpy(spike_train).float()  # (channels, timesteps)
    labels = torch.from_numpy(cursor_pos).float()    # (2, timesteps)

    # Convert position to velocity via gradient
    labels = torch.gradient(labels, dim=1)[0]

    # ========================================
    # Find segment boundaries
    # ========================================
    target_diff = np.diff(target_pos, axis=1, append=target_pos[:, -1:])
    segment_boundaries = np.nonzero(np.sum(np.abs(target_diff), axis=0))[0]

    # Create segment start/end pairs
    segment_boundaries = np.insert(segment_boundaries, 0, 0)
    segment_boundaries = np.append(segment_boundaries, target_pos.shape[1])
    time_segments = np.column_stack([segment_boundaries[:-1], segment_boundaries[1:]])

    # ========================================
    # Split into train/val/test
    # ========================================
    total_segments = time_segments.shape[0]
    sub_length = int(total_segments / split_num)
    stride_steps = int(stride / SAMPLING_RATE)

    train_len = math.floor(train_ratio * sub_length)
    val_len = math.floor((sub_length - train_len) / 2)

    ind_train, ind_val, ind_test = [], [], []

    for split_no in range(split_num):
        for i in range(sub_length):
            seg_idx = split_no * sub_length + i
            if seg_idx >= total_segments:
                break

            seg_start = time_segments[seg_idx, 0]
            seg_end = time_segments[seg_idx, 1]

            if i < train_len:
                ind_train += list(np.arange(seg_start, seg_end, stride_steps))
            elif train_len <= i < train_len + val_len:
                ind_val += list(np.arange(seg_start, seg_end, stride_steps))
            else:
                ind_test += list(np.arange(seg_start, seg_end, stride_steps))

    dataset.close()

    # Determine input size from filename
    if "indy" in filename:
        input_size = 96
    elif "loco" in filename:
        input_size = 192
    else:
        input_size = samples.shape[0]

    print(f"Data loaded: {samples.shape[0]} channels, {samples.shape[1]} timesteps")
    print(f"Split: train={len(ind_train)}, val={len(ind_val)}, test={len(ind_test)}")

    return {
        "samples": samples,
        "labels": labels,
        "ind_train": ind_train,
        "ind_val": ind_val,
        "ind_test": ind_test,
        "ratio": ratio,
        "input_feature_size": input_size,
    }


class PrimateReachingDataset(Dataset):
    """
    PyTorch Dataset for windowed spike data.

    For each index, returns a window of historical timesteps ending at that index.
    This allows the SNN to process temporal context for prediction.
    """

    def __init__(
        self,
        samples: torch.Tensor,
        labels: torch.Tensor,
        indices: list,
        window: int,
        ratio: int,
    ):
        """
        Args:
            samples: Full spike data tensor (channels, total_timesteps)
            labels: Full velocity labels tensor (2, total_timesteps)
            indices: List of valid sample indices for this split
            window: Number of historical timesteps to include
            ratio: Spacing between timesteps (for binning)
        """
        self.samples = samples
        self.labels = labels
        self.indices = indices
        self.window = window
        self.ratio = ratio

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get a windowed sample.

        Returns:
            sample: (channels, window) tensor of spike data
            label: (2, window) tensor of velocity labels
        """
        actual_idx = self.indices[idx]

        # Get window of historical timesteps: [idx-n*ratio, ..., idx-ratio, idx]
        # Oldest first (reversed)
        mask = list(reversed(actual_idx - np.arange(self.window) * self.ratio))

        return self.samples[:, mask], self.labels[:, mask]
