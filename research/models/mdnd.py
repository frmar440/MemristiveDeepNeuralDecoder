"""Define memristive deep neural decoders.
"""
from typing import Optional, OrderedDict, Any

from torch.nn import Sequential
from aihwkit.nn import AnalogRNN, AnalogLinear, AnalogSequential
from aihwkit.nn.modules.base import RPUConfigAlias


class MDND(AnalogSequential):
    """"Memristive deep neural decoder.

    Do not instantiate this class explicitly, call get_mdnd().
    """


def get_mdnd(
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
        weight_scaling_omega: Optional[float] = None
    ) -> MDND:
    """Get memristive deep neural decoder."""

    rnn = AnalogRNN(input_size, hidden_size, num_layers, nonlinearity,
                    bias, batch_first, dropout, bidirectional, rpu_config,
                    realistic_read_write, weight_scaling_omega)
    linear = AnalogLinear(hidden_size, output_size, bias, rpu_config,
                            realistic_read_write, weight_scaling_omega)
    return MDND(rnn, linear)
