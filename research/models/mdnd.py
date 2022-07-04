"""Define memristive deep neural decoders.
"""
from typing import Optional

from torch.nn import Module
from aihwkit.nn import AnalogRNN, AnalogLinear, AnalogSequential
from aihwkit.nn.modules.base import RPUConfigAlias


class MDND(Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            num_layers: int = 1,
            nonlinearity: str = 'relu',
            bias: bool = True,
            batch_first: bool = False,
            dropout: float = 0.,
            bidirectional: bool = False,
            rpu_config: Optional[RPUConfigAlias] = None,
            realistic_read_write: bool = False,
            weight_scaling_omega: Optional[float] = None,
            id: str = None
    ):
        super().__init__()
        self.id = id
        self.rnn_relu_linear_stack = AnalogSequential(
            AnalogRNN(input_size, hidden_size, num_layers, nonlinearity,
                      bias, batch_first, dropout, bidirectional, rpu_config,
                      realistic_read_write, weight_scaling_omega),
            AnalogLinear(hidden_size, output_size, bias, rpu_config,
                         realistic_read_write, weight_scaling_omega)
        )
        

    def forward(self, inputs):  # pylint: disable=arguments-differ
        """Forward pass """
        logits = self.rnn_relu_linear_stack(inputs)
        return logits
