"""Define memristive deep neural decoders.
"""
from typing import Optional, Any, OrderedDict

from torch import Tensor
from aihwkit.exceptions import ModuleError
from aihwkit.nn import AnalogRNN, AnalogLinear, AnalogSequential
from aihwkit.nn.modules.base import RPUConfigAlias, AnalogModuleBase
from aihwkit.simulator.tiles import InferenceTile


class MDND(AnalogSequential):
    """Memristive deep neural decoder.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        output_size: The number of features in the output `y`
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
    ) -> None:
        super().__init__()

        self.rnn = AnalogRNN(input_size, hidden_size, num_layers, nonlinearity,
                   bias, batch_first, dropout, bidirectional, rpu_config,
                   realistic_read_write, weight_scaling_omega)
        self.linear = AnalogLinear(hidden_size, output_size, bias, rpu_config,
                            realistic_read_write, weight_scaling_omega)

    @classmethod
    def from_digital(cls, module,  # pylint: disable=unused-argument
                     *args: Any,
                     **kwargs: Any) -> 'MDND':
        """Construct MDND in-place from DND."""
        analog_module = cls(module.rnn.input_size,
                            module.rnn.hidden_size,
                            module.linear.out_features)
        
        analog_module._modules = module._modules

        return analog_module

    def load_rpu_config(self, rpu_config):
        self._apply_to_analog(lambda m: m._load_from_rpu_config(rpu_config))

    def get_weights(self):
        weights = []
        for module in self.modules():
            if isinstance(module, AnalogModuleBase):
                weights.extend(module.get_weights())
        return weights

    def set_reference_combined_weights(self):
        """Set reference combined weights of all analog inference layers."""
        self._apply_to_analog(lambda m: m.set_reference_combined_weights())

    def get_conductances(self):
        conductances = []
        for module in self.modules():
            if isinstance(module, AnalogModuleBase):
                for analog_tile in module.analog_tiles():
                    if isinstance(analog_tile, InferenceTile):
                        target_conductances, params = analog_tile.noise_model.g_converter.convert_to_conductances(
                            Tensor(analog_tile.tile.get_weights())
                        )
                        conductances.append(target_conductances)
        return conductances                  

    def forward(self, inputs: Tensor):
        """Compute the forward pass."""
        hx = self.rnn(inputs)
        logits = self.linear(hx)
        return logits
