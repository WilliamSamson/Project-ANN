import numpy as np
import pandas as pd
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Paths
training_data_path = '/Training_set/Formatted_Training_Data.csv'
generated_data_path = '/home/kayode-olalere/PycharmProjects/Project ANN/Codebase/Data_Gen/generated_input_dataset.csv'

# Load training dataset
data = pd.read_csv(training_data_path)

# Extract features and targets
X = data[["Frequency (GHz)", "W1 (mm)", "L1 (mm)", "D1 (mm)", "W2 (mm)", "L2 (mm)"]].values
y = data[["S1", "S2"]].values

# Split data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

# Scale input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Build the ANN model
model = Sequential([
    Dense(128, input_dim=X_train_scaled.shape[1], kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),

    Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),

    Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),

    Dense(2)  # Predict S1 and S2
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Define callbacks
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

# Evaluate on test data
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")

# Plot loss curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('your_plot.png')  # Instead of plt.show()

#plt.show()

# Predict outputs for the new dataset
generated_data = pd.read_csv(generated_data_path)
X_generated = generated_data[["Frequency (GHz)", "W1 (mm)", "L1 (mm)", "D1 (mm)", "W2 (mm)", "L2 (mm)"]].values
X_generated_scaled = scaler.transform(X_generated)

# Predict S1 and S2
predictions = model.predict(X_generated_scaled)
generated_data[["S1", "S2"]] = predictions

# Save predictions to CSV
generated_data.to_csv('generated_output_dataset.csv', index=False)
print("Predictions saved to 'generated_output_dataset.csv'.")

# Optional: Retrain the model with new predictions
new_data = pd.concat([data, generated_data])
X_new = new_data[["Frequency (GHz)", "W1 (mm)", "L1 (mm)", "D1 (mm)", "W2 (mm)", "L2 (mm)"]].values
y_new = new_data[["S1", "S2"]].values

X_new_scaled = scaler.fit_transform(X_new)
model.fit(X_new_scaled, y_new, epochs=200, batch_size=32, verbose=2)
