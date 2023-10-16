import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size,num_layers=num_layers, batch_first=True)

        # Fully connected layer to map LSTM output to the desired sequence length (500)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        #h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        #c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        h0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Apply fully connected layer to get the output sequence
        output_seq = self.fc(lstm_out)

        return output_seq

# Rest of the code remains the same as before

