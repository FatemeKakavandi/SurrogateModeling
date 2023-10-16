import torch.optim as optim
import numpy as np
import torch
from model import LSTMModel
import torch.nn as nn
from src_fcn import return_train_test_data,return_x_y, normalization, denormalize
import matplotlib.pyplot as plt
from fsutils import resource_file_path
import pandas as pd
import math
from Load_data import load_set_table
# Assuming you have 17 samples with 7 features each and a target force profile with 500 data points
# X_train: (17, 7) numpy array containing the input features
# y_train: (17, 500) numpy array containing the target force profile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x, y = return_x_y()

X_train, X_test, y_train_denorm, y_test_denorm = return_train_test_data(x, y)

# mapping force profile to the [0,1] range
y_train, y_train_max = normalization(y_train_denorm)
y_test,y_test_max = normalization(y_test_denorm)

#X_train = np.random.randn(17, 7)  # Replace this with your actual input data
#y_train = np.random.randn(17, 500)  # Replace this with your actual target data

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.float32).to(device)
whole_x = torch.tensor(np.array(x), dtype=torch.float32).to(device)

# Initialize the LSTM model
input_size = 7
hidden_size = 250
output_size = 500
epoch_list = [1000,10000,20000,30000,40000,50000]
for num_epochs in epoch_list:
    model = LSTMModel(input_size, hidden_size, output_size,num_layers=3).to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    #num_epochs = 100000
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    X_test_tensor = torch.tensor(np.array(X_test), dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(np.array(y_test), dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        predicted_force_profiles = model(X_test_tensor)

    y_test_pred = predicted_force_profiles.cpu().data.numpy()
    y_test_pred = denormalize(y_test_pred,y_test_max)
    print(num_epochs)
    print(abs(y_test_pred-y_test_denorm).mean())

