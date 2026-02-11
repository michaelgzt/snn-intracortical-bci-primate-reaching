"""
Spiking Neural Network Model

3-layer SNN with Leaky Integrate-and-Fire (LIF) neurons for velocity prediction.
"""

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNN3(nn.Module):
    """
    3-layer Spiking Neural Network for velocity prediction.

    Architecture:
        Input (96 channels for indy, 192 for loco)
        -> fc1 (Linear) -> dropout -> lif1 (LIF, reset='zero') -> 50 neurons
        -> fc2 (Linear) -> dropout -> lif2 (LIF, reset='zero') -> 50 neurons
        -> fc3 (Linear) -> dropout -> lif3 (LIF, reset='zero') -> 50 neurons
        -> fc_out (Linear) -> lif_out (LIF, reset='none') -> 2 neurons

    The output is the membrane potential of the final LIF layer, not spikes.
    This allows smooth regression output for velocity prediction.
    """

    def __init__(
        self,
        input_size: int = 96,
        hidden_size: int = 50,
        output_size: int = 2,
        tau: float = 0.96,
        threshold: float = 1.0,
        dropout_p: float = 0.2,
        device: str = "cpu",
    ):
        """
        Args:
            input_size: Number of input channels (96 for indy, 192 for loco)
            hidden_size: Neurons per hidden layer (default 50)
            output_size: Output dimensions (default 2 for x,y velocity)
            tau: Membrane decay factor (beta) (default 0.96)
            threshold: Spike threshold voltage (default 1.0)
            dropout_p: Dropout probability (default 0.2)
            device: Torch device
        """
        super().__init__()

        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Surrogate gradient for backpropagation through spikes
        spike_grad = surrogate.fast_sigmoid(slope=20)

        # Linear layers (no bias)
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False, device=device)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=False, device=device)
        self.fc3 = nn.Linear(hidden_size, hidden_size, bias=False, device=device)
        self.fc_out = nn.Linear(hidden_size, output_size, bias=False, device=device)

        # LIF neurons for hidden layers (reset to zero on spike)
        lif_params = {
            "beta": tau,
            "spike_grad": spike_grad,
            "threshold": threshold,
            "learn_beta": False,
            "learn_threshold": False,
            "reset_mechanism": "zero",
        }

        self.lif1 = snn.Leaky(**lif_params)
        self.lif2 = snn.Leaky(**lif_params)
        self.lif3 = snn.Leaky(**lif_params)

        # LIF neuron for output (no reset - accumulates for smooth output)
        self.lif_out = snn.Leaky(
            beta=tau,
            spike_grad=spike_grad,
            threshold=threshold,
            learn_beta=False,
            learn_threshold=False,
            reset_mechanism="none",
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_p)

        # Membrane potentials (initialized in reset())
        self.mem1 = None
        self.mem2 = None
        self.mem3 = None
        self.mem_out = None

    def reset(self):
        """Reset membrane potentials to initial state."""
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()
        self.mem3 = self.lif3.init_leaky()
        self.mem_out = self.lif_out.init_leaky()

    def forward(self, x: torch.Tensor, return_spikes: bool = False):
        """
        Forward pass through the SNN.

        Args:
            x: Input tensor of shape (batch, channels, window)
            return_spikes: If True, return spike activity per layer

        Returns:
            mem_rec: Output membrane potentials, shape (batch, 2, window)
            spike_activity: Array of mean spike rates per layer (if return_spikes)
        """
        window = x.shape[2]

        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem_out = self.lif_out.init_leaky()

        mem_rec = []
        spike_activity = x.mean().cpu().detach().numpy()

        # Process each timestep
        for step in range(window):
            # Layer 1
            cur1 = self.dropout(self.fc1(x[:, :, step]))
            spk1, mem1 = self.lif1(cur1, mem1)
            spike_activity1 = spk1.mean().cpu().detach().numpy()

            # Layer 2
            cur2 = self.dropout(self.fc2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            spike_activity2 = spk2.mean().cpu().detach().numpy()

            # Layer 3
            cur3 = self.dropout(self.fc3(spk2))
            spk3, mem3 = self.lif3(cur3, mem3)
            spike_activity3 = spk3.mean().cpu().detach().numpy()

            # Output layer (no dropout)
            cur_out = self.fc_out(spk3)
            _, mem_out = self.lif_out(cur_out, mem_out)

            mem_rec.append(mem_out)

        result = torch.stack(mem_rec, dim=2)

        if return_spikes:
            return result, np.array([
                spike_activity, spike_activity1, spike_activity2, spike_activity3
            ])
        return result

    def forward_with_captures(self, x: torch.Tensor) -> dict:
        """
        Forward pass capturing intermediate activations for metrics.

        Args:
            x: Input tensor of shape (batch, channels, window)

        Returns:
            Dictionary with predictions and captured activations
        """
        window = x.shape[2]

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem_out = self.lif_out.init_leaky()

        mem_rec = []
        all_spk1, all_spk2, all_spk3 = [], [], []
        all_inp_fc1, all_inp_fc2, all_inp_fc3, all_inp_fc_out = [], [], [], []

        for step in range(window):
            inp = x[:, :, step]
            all_inp_fc1.append(inp.clone())

            # Layer 1
            cur1 = self.fc1(inp)
            spk1, mem1 = self.lif1(cur1, mem1)
            all_spk1.append(spk1.clone())
            all_inp_fc2.append(spk1.clone())

            # Layer 2
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            all_spk2.append(spk2.clone())
            all_inp_fc3.append(spk2.clone())

            # Layer 3
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            all_spk3.append(spk3.clone())
            all_inp_fc_out.append(spk3.clone())

            # Output
            cur_out = self.fc_out(spk3)
            _, mem_out = self.lif_out(cur_out, mem_out)
            mem_rec.append(mem_out)

        return {
            "predictions": torch.stack(mem_rec, dim=2),
            "all_spk1": all_spk1,
            "all_spk2": all_spk2,
            "all_spk3": all_spk3,
            "all_inp_fc1": all_inp_fc1,
            "all_inp_fc2": all_inp_fc2,
            "all_inp_fc3": all_inp_fc3,
            "all_inp_fc_out": all_inp_fc_out,
        }
