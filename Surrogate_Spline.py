import numpy as np
from pygam import LinearGAM
import joblib
import matplotlib.pyplot as plt
# Generate synthetic data
np.random.seed(0)
n_samples = 17
n_features = 7
n_time_points = 500

# Create an array for the input features with shape (n_samples * n_time_points, n_features)
X = np.random.rand(n_samples * n_time_points, n_features)

# Create an array for timestamps (time points) with shape (n_samples * n_time_points, 1)
timestamps = np.linspace(0, 10, n_samples * n_time_points).reshape(-1, 1)

# Combine input features and timestamps
X = np.hstack((X, timestamps))

# Create an array for the target data with shape (n_samples * n_time_points,)
y = np.random.rand(n_samples * n_time_points)

# Create a LinearGAM model with P-spline functions
gam = LinearGAM(n_splines=25, spline_order=3, lam=1, fit_intercept=True)

# Fit the model to the data
gam.gridsearch(X, y)

# Save the trained model to a file
model_filename = 'pspline_model.pkl'
joblib.dump(gam, model_filename)

# Load the saved model from the file
loaded_model = joblib.load(model_filename)

# Generate full time series for each input X_new with timestamps
X_new = np.random.rand(1, n_features)  # New input data for prediction
timestamps_new = np.linspace(0, 10, n_time_points).reshape(-1, 1)  # Adjust timestamps as needed

y_full_series = []

for i in range(n_time_points):
    X_combined = np.hstack((X_new, timestamps_new[i].reshape(1, 1)))
    y_pred = loaded_model.predict(X_combined)
    y_full_series.append(y_pred)
    #X_new = np.concatenate((X_new, y_pred.reshape(1, 1)), axis=1)

# y_full_series now contains the full time series for the given input X_new with timestamps
plt.plot(y_full_series)
plt.show()