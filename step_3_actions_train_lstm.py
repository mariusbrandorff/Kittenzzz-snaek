"""Task: adapt your RNN to use nn.LSTM.

The LSTM returns outputs, (hidden, cell). We can still classify outputs[:, -1, :].
Keep the input features, hidden width, output shape and training budget the same.
Explain what the additional cell state is for. Is improvement guaranteed here?
"""
from torch import nn


class StudentLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, future):
        super().__init__()
        # TODO: the same small architecture, using an LSTM.
        raise NotImplementedError('Define your LSTM layers in Step 3.')

    def forward(self, observations):
        # TODO: return (batch, future, 4) logits.
        raise NotImplementedError('Implement your LSTM forward pass in Step 3.')


ActionLSTM = StudentLSTM

if __name__ == '__main__':
    from supplied_pipeline import train_cli
    train_cli('lstm', ActionLSTM)
