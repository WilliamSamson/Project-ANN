import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# File paths
training_data_path = '/home/kayode-olalere/PycharmProjects/Project ANN/Model/Formatted_Training_Data.csv'
generated_data_path = '/home/kayode-olalere/PycharmProjects/Project ANN/Codebase/Data_Gen/generated_input_dataset.csv'

# Load training dataset
data = pd.read_csv(training_data_path)

# Extract features and targets
X = data[["Frequency (GHz)", "W1 (mm)", "L1 (mm)", "D1 (mm)", "W2 (mm)", "L2 (mm)"]].values.astype('float32')
y = data[["S1", "S2"]].values.astype('float32')

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
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    BatchNormalization(),
    LeakyReLU(negative_slope=0.1),
    Dropout(0.5),
    Dense(2)  # Predict S1 and S2
])

# Compile the model with advanced loss function
model.compile(optimizer=Adam(learning_rate=0.001), loss=tf.keras.losses.Huber())

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
model_checkpoint = ModelCheckpoint('best_model_advanced.h5', save_best_only=True, monitor='val_loss', verbose=1)

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=1000, batch_size=16,
    validation_data=(X_val_scaled, y_val),
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    verbose=2
)

# Load the best model
model = tf.keras.models.load_model('best_model_advanced.h5')

# Evaluate on test data
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")

# Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test[:, 0], y_pred[:, 0], label='S1 Prediction vs Actual', alpha=0.7)
plt.scatter(y_test[:, 1], y_pred[:, 1], label='S2 Prediction vs Actual', alpha=0.7)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.title('Predictions vs Actual Values')
plt.savefig('predictions_vs_actual.png')

# Predict outputs for new dataset
generated_data = pd.read_csv(generated_data_path)
X_generated = generated_data[["Frequency (GHz)", "W1 (mm)", "L1 (mm)", "D1 (mm)", "W2 (mm)", "L2 (mm)"]].values.astype('float32')
X_generated_scaled = scaler.transform(X_generated)  # Use same scaler as during training

# Predict S1 and S2
predictions = model.predict(X_generated_scaled)
generated_data[["S1", "S2"]] = predictions

# Save predictions to CSV
generated_data.to_csv('generated_output_dataset_advanced.csv', index=False)
print("Predictions saved to 'generated_output_dataset_advanced.csv'.")

# Optional: K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scores = []
r2_scores = []

for train_index, test_index in kf.split(X):
    X_train_kf, X_test_kf = X[train_index], X[test_index]
    y_train_kf, y_test_kf = y[train_index], y[test_index]

    X_train_kf_scaled = scaler.fit_transform(X_train_kf)
    X_test_kf_scaled = scaler.transform(X_test_kf)

    model.fit(
        X_train_kf_scaled, y_train_kf,
        epochs=100, batch_size=16, verbose=0
    )

    y_pred_kf = model.predict(X_test_kf_scaled)
    mse_scores.append(mean_squared_error(y_test_kf, y_pred_kf))
    r2_scores.append(r2_score(y_test_kf, y_pred_kf))

print(f"Average MSE (K-Fold): {np.mean(mse_scores)}")
print(f"Average R2 (K-Fold): {np.mean(r2_scores)}")

