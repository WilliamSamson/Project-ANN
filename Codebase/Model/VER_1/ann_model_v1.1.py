import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
from tensorflow.keras.regularizers import l2

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the dataset
data = pd.read_csv(
    '/Training_set/Formatted_Training_Data.csv')  # Adjust path

# Extract input features (W1, L1, Frequency, etc.)
X = data[["Frequency (GHz)", "W1 (mm)", "L1 (mm)", "D1 (mm)", "W2 (mm)", "L2 (mm)"]].values

# Generate synthetic target values (S1, S2) using refined formulas
f = X[:, 0]
W1 = X[:, 1]
L1 = X[:, 2]
D1 = X[:, 3]
W2 = X[:, 4]
L2 = X[:, 5]

# Define coefficients for the formulas
k1, k2, k3, k4 = 10, -5, 2, 15  # Coefficients for S1
k5, k6, k7, k8 = -60, 3, 1.5, 20  # Coefficients for S2

S1 = (
    k1 * f
    + k2 / np.sqrt(W1 * L1)
    + k3 * D1
    + k4
    + np.random.normal(0, 0.1, size=f.shape)  # Add random noise
)

S2 = (
    k5 * f
    + k6 * W1
    + k7 * L2
    + k8
    + np.random.normal(0, 0.1, size=f.shape)  # Add random noise
)

y = np.stack((S1, S2), axis=1)  # Combine targets (S1, S2) for regression task

# Split dataset into training, testing, and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

# Normalize the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Build the advanced ANN model with L2 regularization and LeakyReLU
model = Sequential([
    Dense(128, input_dim=X_train_scaled.shape[1], activation=None, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),

    Dense(256, activation=None, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),

    Dense(128, activation=None, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),

    Dense(2)  # Output layer for S1 and S2
])

# Compile the model with an improved learning rate schedule
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Define callbacks for optimization and monitoring
log_dir = os.path.join("../logs", "fit")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6, verbose=1)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', verbose=1)

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=500, batch_size=32,
    validation_data=(X_val_scaled, y_val),
    callbacks=[early_stopping, reduce_lr, model_checkpoint, tensorboard_callback],
    verbose=2
)

# Load the best model
model = tf.keras.models.load_model('best_model.h5')

# Evaluate the model on the test set
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display evaluation metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")

# Plot the training and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
