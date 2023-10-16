from fsutils import resource_file_path
import pandas as pd
import matplotlib.pyplot as plt
from model import LSTMModel
import torch
from src_fcn import return_x_y, load_new_input_samples
import torch.nn as nn
import numpy as np
import matplotlib as mpl
mpl.use('macosx')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get input data
input_data = np.array(load_new_input_samples())

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
X_all_data_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)

# Loss function
criterion = nn.MSELoss()

# Generate predictions for all data
with torch.no_grad():
    predicted_force_profiles = loaded_model(X_all_data_tensor)

y_test_pred = predicted_force_profiles.cpu().data.numpy()

# Plotting the synthetic samples with timestamp
t0 = 0
tend = 0.0025
tstep = 0.0025/500

t_temp = list(np.arange(0,1000,2))


for i in range(len(y_test_pred)):
    plt.plot(t_temp, y_test_pred[i])
plt.ylabel('Force(N)',fontsize=12)
plt.xlabel('Index',fontsize=12)

plt.savefig('LSTM_Synth.pdf')
plt.show()

