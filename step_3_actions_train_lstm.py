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
        #raise NotImplementedError('Define your LSTM layers in Step 3.')

        self.future = future
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, future*4)

    def forward(self, observations):
        # TODO: return (batch, future, 4) logits.
        #raise NotImplementedError('Implement your LSTM forward pass in Step 3.')
        
        outputs, self.hidden = self.lstm(observations)
        prediction = self.output(outputs[:, -1, :])
        return prediction.reshape(observations.shape[0], self.future, 4)


ActionLSTM = StudentLSTM

if __name__ == '__main__':
    from supplied_pipeline import train_cli
    train_cli('lstm', ActionLSTM)
