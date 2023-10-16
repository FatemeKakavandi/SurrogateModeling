import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        # Fully connected layer to map LSTM output to the desired sequence length (500)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Apply fully connected layer to get the output sequence
        output_seq = self.fc(lstm_out)

        return output_seq

# Assuming you have 17 samples with 7 features each and a target force profile with 500 data points
# X_train: (17, 7) numpy array containing the input features
# y_train: (17, 500) numpy array containing the target force profile

# Sample data (replace these arrays with your actual data)
X_train = np.random.randn(17, 7)
y_train = np.random.randn(17, 500)

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Initialize the LSTM model
input_size = 7
hidden_size = 64
output_size = 500
model = LSTMModel(input_size, hidden_size, output_size)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# After training, you can use the model to generate force profiles for new input data
# For example, to generate force profiles for new data X_test, you can do:
X_test = np.random.randn(5, 7)  # Replace this with your actual test data
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    predicted_force_profiles = model(X_test_tensor)
