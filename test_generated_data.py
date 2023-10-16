from fsutils import resource_file_path
import pandas as pd
import matplotlib.pyplot as plt
from model import LSTMModel
import torch
from src_fcn import return_x_y
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get all data
x,y = return_x_y()
# Assuming the trained model is saved to 'lstm_model.pth'
# Initialize a new instance of the LSTM model with the same architecture as the saved model
# Initialize the LSTM model
input_size = 7
hidden_size = 250
output_size = 500
loaded_model = LSTMModel(input_size, hidden_size, output_size,num_layers=3).to(device)


# Load the model's state dictionary from the file
loaded_model.load_state_dict(torch.load('lstm_model.pth',map_location=torch.device('cpu')))

# Set the model to evaluation mode (important for models with dropout or batch normalization)
loaded_model.eval()

# Convert all data to PyTorch tensors and move to the same device as the loaded model
X_all_data_tensor = torch.tensor(x, dtype=torch.float32).to(device)
y_all_data_tensor = torch.tensor(y, dtype=torch.float32).to(device)

# Loss function
criterion = nn.MSELoss()

# Generate predictions for all data
with torch.no_grad():
    predicted_force_profiles = loaded_model(X_all_data_tensor)

# Calculate the Mean Squared Error (MSE) between predicted and actual force profiles
mse_loss = criterion(predicted_force_profiles, y_all_data_tensor)

print(f'Mean Squared Error (MSE) on all data: {mse_loss.item():.4f}')

