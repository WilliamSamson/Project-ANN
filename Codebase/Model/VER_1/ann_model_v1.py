import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the dataset
data = pd.read_csv('/Codebase/Data_Gen/generated_input_dataset.csv')  # Adjust path

# Extract input features (W1, L1, Frequency, etc.)
X = data[["Frequency (GHz)", "W1 (mm)", "L1 (mm)", "D1 (mm)", "W2 (mm)", "L2 (mm)"]].values

# Generate synthetic target values (S1, S2)
# Generate synthetic target values (S1, S2)
import numpy as np

# Example input feature matrix X
# Columns: Frequency (f), W1, L1, D1, W2, L2
X = np.random.uniform(0.1, 5.0, size=(100, 6))  # Generate random data for 100 samples

f = X[:, 0]  # Frequency
W1 = X[:, 1]  # Width 1
L1 = X[:, 2]  # Length 1
D1 = X[:, 3]  # Gap (D1)
W2 = X[:, 4]  # Width 2
L2 = X[:, 5]  # Length 2

# Coefficients for the synthetic formula
k1, k2, k3, k4, k5, k6, k7 = 0.5, 3.0, 0.8, 2.5, 1.2, 0.1, 5.0

# Calculate S1 and S2 using the proposed formula
S1 = (
    k1 * f +
    k2 * (1 / np.sqrt(W1 * L1)) +
    k3 * D1 +
    k4 * (1 / W2) +
    k5 * L2 +
    k6 * f * (D1 / W1) +
    k7 +
    np.random.normal(0, 0.1, size=f.shape)  # Add random noise
)

S2 = (
    -60 +
    k4 * np.log(f + 1) +
    k3 * D1 -
    np.random.uniform(0.1, 3.0, size=f.shape) +  # Random substrate thickness effect
    np.random.normal(0, 0.1, size=f.shape)  # Add random noise
)

# Combine targets (S1, S2) for regression task
y = np.stack((S1, S2), axis=1)

# Split dataset into training, testing, and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

# Normalize the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Build the optimized ANN model
model = Sequential([
    Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(2)  # Output layer for S1 and S2
])

# Compile the model with dynamic learning rate
model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

# Define callbacks for optimization
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6, verbose=1)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', verbose=1)

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=500, batch_size=32,
    validation_data=(X_val_scaled, y_val),
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
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
