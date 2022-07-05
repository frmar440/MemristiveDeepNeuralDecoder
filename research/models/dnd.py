"""Define digital neural network models.
"""
from torch.nn import Module, RNN, Linear


class DND(Module):
    """Deep neural decoder.

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
        bidirectional: bool = False
    ) -> None:
        super().__init__()
        
        self.rnn = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                       nonlinearity=nonlinearity, bias=bias, batch_first=batch_first,
                       dropout=dropout, bidirectional=bidirectional)
        self.linear = Linear(hidden_size, output_size, bias)

    def forward(self, X):
        """Implement the operations on input data. Do not call this method directly.
        To use the model, pass it the input data (e.g. model(X)): this executes
        forward and some background operations.
        Args:
            X (torch.Tensor): tensor of shape (batch_size, sequence_length, input_size) when batch_first=True.
        Returns:
            torch.Tensor: raw predicted values for each class.
        """
        output, _ = self.rnn(X)
        logits = self.linear(output[-1, :, :]) # output features h_t for the last timestep
        return logits
