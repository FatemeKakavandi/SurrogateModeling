import torch.optim as optim
import numpy as np
import torch
from model import LSTMModel
import torch.nn as nn
from src_fcn import return_train_test_data,return_x_y
from fsutils import resource_file_path
import pandas as pd
from Load_data import load_set_table
# Assuming you have 17 samples with 7 features each and a target force profile with 500 data points
# X_train: (17, 7) numpy array containing the input features
# y_train: (17, 500) numpy array containing the target force profile

device = 'cuda:0'
x,y = return_x_y()
X_train, X_test, y_train, y_test = return_train_test_data(x,y)

#X_train = np.random.randn(17, 7)  # Replace this with your actual input data
#y_train = np.random.randn(17, 500)  # Replace this with your actual target data

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.float32).to(device)

# Initialize the LSTM model
input_size = 7
hidden_size = 250
output_size = 500
model = LSTMModel(input_size, hidden_size, output_size,num_layers=3).to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 10000
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
  # Replace this with your actual test data
X_test_tensor = torch.tensor(np.array(X_test), dtype=torch.float32).to(device)
with torch.no_grad():
    predicted_force_profiles = model(X_test_tensor)

test_df = pd.DataFrame(predicted_force_profiles.cpu().data.numpy())
test_df.to_csv('output_data.csv')