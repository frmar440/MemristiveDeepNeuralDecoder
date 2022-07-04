"""Define memristive deep neural decoders.
"""
from typing import Optional, OrderedDict, Any

from torch.nn import Sequential
from aihwkit.nn import AnalogRNN, AnalogLinear, AnalogSequential
from aihwkit.nn.modules.base import RPUConfigAlias


class MDND(Sequential):
    """"Memristive deep neural decoder.
    """

    @classmethod
    def from_digital(cls, module,  # pylint: disable=unused-argument
                     *args: Any,
                     **kwargs: Any) -> 'MDND':
        """Construct MDND in-place from DND."""
        return cls(OrderedDict(mod for mod in module.named_children()))
    
    def forward(self, inputs):  # pylint: disable=arguments-differ
        """Forward pass """
        rnn, linear = self
        hx = rnn(inputs)
        logits = linear(hx)
        return logits

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
        weight_scaling_omega: Optional[float] = None,
        id: str = None
    ) -> MDND:
    rnn = AnalogRNN(input_size, hidden_size, num_layers, nonlinearity,
                    bias, batch_first, dropout, bidirectional, rpu_config,
                    realistic_read_write, weight_scaling_omega)
    linear = AnalogLinear(hidden_size, output_size, bias, rpu_config,
                            realistic_read_write, weight_scaling_omega)
    return MDND(rnn, linear)
