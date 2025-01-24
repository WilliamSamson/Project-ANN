

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
data = pd.read_csv('/home/kayode-olalere/PycharmProjects/Project ANN/Codebase/Data_Gen/generated_input_dataset.csv')  # Adjust with your actual file path

# Extract input features (W1, L1, Frequency, etc.)
X = data[["Frequency (GHz)", "W1 (mm)", "L1 (mm)", "D1 (mm)", "W2 (mm)", "L2 (mm)"]].values

# Generate synthetic target values (S1, S2)
# These formulas are based on the ones you provided earlier
f = X[:, 0]
W1 = X[:, 1]
L1 = X[:, 2]
W2 = X[:, 4]
substrate_thickness = np.random.uniform(0.1, 3.0, size=f.shape)  # Example for random substrate thickness

S1 = 10 - (W1 + L1) * f + np.random.normal(0, 0.1, size=f.shape)  # Add random noise
S2 = -60 + (W2 * np.log(f + 1)) - substrate_thickness + np.random.normal(0, 0.1, size=f.shape)  # Add random noise

# Combine the targets (S1, S2) into a single array (for regression task)
y = np.stack((S1, S2), axis=1)

# Split dataset into training, testing, and validation sets (70%, 20%, 10%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

# Normalize the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Build the ANN model
model = Sequential()
model.add(Dense(15, input_dim=X_train_scaled.shape[1], activation='tanh'))  # 1st hidden layer
model.add(Dense(15, activation='tanh'))  # 2nd hidden layer
model.add(Dense(15, activation='tanh'))  # 3rd hidden layer
model.add(Dense(15, activation='tanh'))  # 4th hidden layer
model.add(Dense(2))  # Output layer (S1 and S2)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.5), loss='mean_squared_error')

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=3000, batch_size=64, validation_data=(X_val_scaled, y_val), verbose=2)

# Evaluate the model on the test set
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display evaluation metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")

# Optionally, plot the training and validation loss over epochs
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()