# SNN Training for Primate Reaching Velocity Prediction

This codebase implements a 3-layer Spiking Neural Network (SNN) for predicting finger velocity from neural spike data, based on the approach described in:

> **"Benchmarking spiking neural network learning methods with varying locality"**
> Paul Hueber et al., Benchmarking of hardware-efficient real-time neural decoding in brain–computer interfaces, Neuromorphic Computing and Engineering, 2024, https://iopscience.iop.org/article/10.1088/2634-4386/ad4411/meta

## Key Differences from the Paper

| Aspect | Paper | This Implementation |
|--------|-------|---------------------|
| **Data Split** | K-fold cross-validation | Train/Val/Test split (same as NeuroBench: 50% train, 25% val, 25% test) |
| **Evaluation** | Average across folds | Single test set evaluation |
| **Framework** | Custom code | Refactored with NeuroBench compatibility |

The core SNN architecture (3-layer LIF network with surrogate gradient training) remains the same.

## Project Structure

```
refactored_snn_training_testing/
├── model.py              # SNN3 architecture definition
├── config.py             # Configuration dataclass with hyperparameters
├── dataset.py            # Dataset loading and preprocessing
├── train.py              # Training loop with early stopping
├── test.py               # Testing and benchmark metrics
├── loss.py               # Temporal weighted MSE loss
├── metrics.py            # R2, sparsity, synaptic operations metrics
├── download.py           # Dataset download from Zenodo
├── main.py               # CLI entry point for training
├── evaluate.py           # CLI entry point for evaluation
├── neurobench_test.py    # NeuroBench framework benchmarking
├── __init__.py           # Package exports
├── data/                 # Dataset files (.mat)
└── best_model_*.pt       # Trained model checkpoints
```

## Code Files Description

### Core Modules

| File | Description |
|------|-------------|
| **model.py** | Defines `SNN3` - a 3-layer Spiking Neural Network using snntorch. Architecture: Input → FC(50) → LIF → FC(50) → LIF → FC(50) → LIF → FC(2) → LIF. Uses surrogate gradient (fast sigmoid) for backpropagation through spikes. |
| **config.py** | `SNNConfig` dataclass containing all hyperparameters (learning rate, batch size, tau, threshold, window size, etc.). Auto-detects input size from filename (96 for indy, 192 for loco). |
| **dataset.py** | Loads primate reaching data from HDF5 .mat files. Converts spike times to binary spike trains, computes velocity from cursor position, and creates windowed training samples. |
| **train.py** | Training loop with AdamW optimizer, temporal weighted MSE loss, validation-based early stopping, and checkpoint saving. |
| **test.py** | Inference utilities and comprehensive benchmarking. Computes R2 score, footprint, connection sparsity, activation sparsity, and synaptic operations. |
| **loss.py** | Temporal weighted MSE loss - weights later timesteps more heavily since SNN membrane potentials need time to stabilize. |
| **metrics.py** | All evaluation metrics: footprint, connection sparsity, R2, Pearson correlation, MSE, activation sparsity, synaptic operations (MACs/ACs). |
| **download.py** | Downloads dataset from Zenodo with MD5 checksum verification. |

### CLI Entry Points

| File | Description |
|------|-------------|
| **main.py** | Command-line interface for training. Parses arguments and invokes training loop. |
| **evaluate.py** | Command-line interface for evaluation. Loads trained model and runs benchmarks. |
| **neurobench_test.py** | Benchmarks trained models using the [NeuroBench](https://github.com/NeuroBench/neurobench) framework with standardized metrics. |

## Available Sessions

The dataset contains recordings from two non-human primates (NHP):

| Session | Primate | Channels | File Size |
|---------|---------|----------|-----------|
| `indy_20160622_01` | Indy | 96 | 909 MB |
| `indy_20160630_01` | Indy | 96 | 382 MB |
| `indy_20170131_02` | Indy | 96 | 209 MB |
| `loco_20170301_05` | Loco | 192 | - |
| `loco_20170215_02` | Loco | 192 | - |
| `loco_20170210_03` | Loco | 192 | - |

## Quick Start

### 1. Environment Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Please refer to the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) to install uv first.

```bash
# Clone and set up
git clone https://github.com/guangzhi-qu/snn-intracortical-bci-primate-reaching.git
cd snn-intracortical-bci-primate-reaching

# Create virtual environment and install dependencies
uv sync
```

Run commands from the uv environment:

```bash
# Using uv run
uv run python main.py --data-dir ./data --download

# Or activate the virtual environment first
source .venv/bin/activate
python main.py --data-dir ./data --download
```

### 2. Training

Train on a specific session:

```bash
# Train on indy_20160622_01 (default)
python main.py --data-dir ./data --download

# Train on a different session
python main.py --data-dir ./data --filename indy_20160630_01.mat --download

# Train on loco session (192 input channels)
python main.py --data-dir ./data --filename loco_20170301_05.mat --download

# Custom hyperparameters
python main.py --data-dir ./data \
    --filename indy_20160622_01.mat \
    --epochs 100 \
    --batch-size 256 \
    --lr 0.001 \
    --hidden-size 50 \
    --dropout 0.2 \
    --window 50 \
    --patience 20 \
    --save-dir ./checkpoints
```

**Training Output:**
- Model checkpoint saved as `best_model_{filename}.pt`
- Includes model weights and test metrics

### 3. Evaluation

Evaluate a trained model:

```bash
# Evaluate on the same session used for training
python evaluate.py --model best_model_indy_20160622_01_42.pt \
    --data-dir ./data \
    --filename indy_20160622_01.mat

# Evaluate with specific model parameters (must match training)
python evaluate.py --model best_model_indy_20160630_01_42.pt \
    --data-dir ./data \
    --filename indy_20160630_01.mat \
    --hidden-size 50 \
    --dropout 0.2 \
    --window 50
```

**Evaluation Output:**
```
R2 Score:             0.6874
MSE:                  0.0012
Footprint:            39680 bytes
Connection Sparsity:  0.0000
Activation Sparsity:  0.9772
Synaptic Operations:
  Effective MACs:     0.00
  Effective ACs:      549.03
  Dense:              9900.00
```

### 4. NeuroBench Benchmarking

Run standardized benchmarks using the NeuroBench framework:

```bash
# Benchmark all trained models
python neurobench_test.py
```

**NeuroBench Output:**
```
============================================================
NeuroBench Results for indy_20160622_01
============================================================
Static Metrics:
  Footprint:            39,680 bytes
  Connection Sparsity:  0.0000

Workload Metrics:
  R2 Score:             0.687392
  Activation Sparsity:  0.977207

Synaptic Operations:
  Effective MACs:       0.00
  Effective ACs:        549.03
  Dense:                9900.00
============================================================
```

To benchmark a specific session, modify `neurobench_test.py` or use:

```python
from neurobench_test import run_neurobench_benchmark

results = run_neurobench_benchmark(
    model_path="best_model_indy_20160622_01.pt",
    data_dir="./data",
    filename="indy_20160622_01",
)
print(results)
```

## Training All Sessions

To train and evaluate on all available sessions:

```bash
# Train all indy sessions
for session in indy_20160622_01 indy_20160630_01 indy_20170131_02; do
    python main.py --data-dir ./data --filename ${session}.mat --download
done

# Train all loco sessions
for session in loco_20170301_05 loco_20170215_02 loco_20170210_03; do
    python main.py --data-dir ./data --filename ${session}.mat --download
done
```

## Model Architecture

```
Input (96/192 channels, 50 timesteps)
    │
    ▼
┌─────────────────────────────────────┐
│  FC Layer 1 (input → 50, no bias)   │
│  Dropout (p=0.2)                    │
│  LIF Neuron (β=0.96, reset='zero')  │
└─────────────────────────────────────┘
    │ spikes
    ▼
┌─────────────────────────────────────┐
│  FC Layer 2 (50 → 50, no bias)      │
│  Dropout (p=0.2)                    │
│  LIF Neuron (β=0.96, reset='zero')  │
└─────────────────────────────────────┘
    │ spikes
    ▼
┌─────────────────────────────────────┐
│  FC Layer 3 (50 → 50, no bias)      │
│  Dropout (p=0.2)                    │
│  LIF Neuron (β=0.96, reset='zero')  │
└─────────────────────────────────────┘
    │ spikes
    ▼
┌─────────────────────────────────────┐
│  FC Output (50 → 2, no bias)        │
│  LIF Neuron (β=0.96, reset='none')  │
└─────────────────────────────────────┘
    │ membrane potential
    ▼
Output (2D velocity: x, y)
```

**Key Design Choices:**
- **Surrogate Gradient:** Fast sigmoid (slope=20) for backpropagation through spikes
- **Output Layer:** Uses `reset='none'` to accumulate information for smooth regression output
- **Temporal Weighted Loss:** Early timesteps weighted less (0→1 linear schedule)

## Benchmark Results

| Session | R2 Score | Activation Sparsity | Effective ACs |
|---------|----------|---------------------|---------------|
| indy_20160622_01 | 0.687 | 97.7% | 549.03 |
| indy_20160630_01 | 0.452 | 99.2% | 293.01 |
| indy_20170131_02 | 0.551 | 97.3% | 361.28 |
| **Average** | **0.564** | **98.1%** | **401.11** |

## License

This code is provided for research purposes. Please cite the original paper if you use this code.
