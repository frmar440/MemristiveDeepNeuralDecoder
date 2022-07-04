""" Recurrent layers."""

import math
import warnings
import numbers
from typing import List, Tuple, Optional, overload

import torch
from torch import nn
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.rnn import RNNBase, RNN

from aihwkit.nn import AnalogSequential
from aihwkit.nn import AnalogLinear
from aihwkit.nn.modules.base import AnalogModuleBase, RPUConfigAlias
from aihwkit.simulator.configs import SingleRPUConfig


class AnalogRNN(Module):
    """RNN layer that uses an analog tile.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two RNNs together to form a `stacked RNN`,
            with the second RNN taking in outputs of the first RNN and
            computing the final results. Default: 1
        nonlinearity: The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``. Default: ``'tanh'``
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Note that this does not apply to hidden or cell states. See the
            Inputs/Outputs sections below for details.  Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            RNN layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``
        rpu_config: resistive processing unit configuration.
        realistic_read_write: whether to enable realistic read/write
            for setting initial weights and during reading of the weights.
        weight_scaling_omega: The weight scaling omega factor (see
            :class:`~aihwkit.configs.utils.MappingParameter`). If
            given explicitly here, it will overwrite the value in
            the mapping field.
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            nonlinearity: str = 'tanh',
            bias: bool = True,
            batch_first: bool = False,
            dropout: float = 0.,
            bidirectional: bool = False,
            rpu_config: Optional[RPUConfigAlias] = None,
            realistic_read_write: bool = False,
            weight_scaling_omega: Optional[float] = None
    ):
        super().__init__()
        if num_layers != 1:
            raise ValueError('Only one layer is supported')
        if bidirectional:
            raise ValueError('Bidirectional RNN is not supported')

        if nonlinearity == 'tanh':
            activation = nn.Tanh()
        elif nonlinearity == 'relu':
            activation = nn.ReLU()
        else:
            raise ValueError("Unknown nonlinearity '{}'".format(self.nonlinearity))

        self.batch_first = batch_first
        self.hidden_size = hidden_size

        in_features = input_size + hidden_size
        out_features = hidden_size

        self.rnn = AnalogSequential(
            AnalogLinear(in_features, out_features, bias, rpu_config,
                         realistic_read_write, weight_scaling_omega),
            activation
        )

    @classmethod
    def from_digital(
            cls,
            module: RNN,
            rpu_config: Optional[RPUConfigAlias] = None,
            realistic_read_write: bool = False
    ) -> 'AnalogRNN':
        """Return an AnalogRNN layer from a torch RNN layer."""
        analog_module = cls(module.input_size,
                            module.hidden_size,
                            module.num_layers,
                            module.nonlinearity,
                            module.bias,
                            module.batch_first,
                            module.dropout,
                            module.bidirectional,
                            rpu_config,
                            realistic_read_write)

        weight = torch.cat((module.weight_ih_l0, module.weight_hh_l0), dim=1)
        bias = torch.add(module.bias_ih_l0, module.bias_hh_l0)

        analog_module.rnn._apply_to_analog(lambda m: m.set_weights(weight, bias))
        return analog_module

    def reset_parameters(self) -> None:
        """Reset the parameters (weight and bias)."""
        self.rnn._apply_to_analog(lambda m: m.reset_parameters())

    def forward(self, inputs: Tensor):
        """Compute the forward pass."""
        is_batched = inputs.dim() == 3
        batch_dim = 0 if self.batch_first else 1
        if not is_batched:
            inputs = inputs.unsqueeze(batch_dim)
        max_batch_size = inputs.size(0) if self.batch_first else inputs.size(1)

        hx = torch.zeros(max_batch_size, self.hidden_size,
                         dtype=inputs.dtype, device=inputs.device)
        
        if self.batch_first:
            inputs = torch.transpose(inputs, 0, 1) # swap batch_size and sequence_length
        
        for input in inputs: # iterate through sequence
            x_input = torch.cat((input, hx), dim=1)
            hx = self.rnn(x_input)
        
        return hx
