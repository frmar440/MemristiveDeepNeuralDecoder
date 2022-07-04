"""Define digital neural network models.
"""
from torch.nn import Module, RNN, Linear


class DND(Module):
    """Deep neural decoder.
    """
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        """Initialize neural network stack.

        Args:
            input_size (int): number of features in input.
            hidden_size (int, optional): number of features in hidden. Defaults to 128.
            num_layers (int, optional): number of recurrent layers. Defaults to 1.
        """
        super().__init__()
        
        self.rnn = RNN(input_size, hidden_size, num_layers, nonlinearity='relu')
        self.linear = Linear(hidden_size, output_size)

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
