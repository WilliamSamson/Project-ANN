import os
import warnings
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input, LeakyReLU, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import mlflow
import joblib
from pathlib import Path


# ========================
# Configuration Parameters
# ========================
class Config:
    SEED = 42
    TEST_SIZE = 0.20
    VAL_SIZE = 0.10
    EPOCHS = 800
    BATCH_SIZE = 128
    EARLY_STOPPING_PATIENCE = 15
    REDUCE_LR_PATIENCE = 7
    REGULARIZATION = l1_l2(l1=1e-5, l2=1e-4)
    FEATURE_COLS = ["l_s", "l_2", "l_1", "s_2", "s_1", "w_s", "w_2", "w_1", "freq"]
    TARGET_COLS = ["dB(S(1,1))", "dB(S(2,1))"]
    CHECKPOINT_PATH = "best_model.h5"
    INITIAL_LR = 0.001
    CLIP_VALUE = 1.0  # Gradient clipping threshold


# ========================
# Residual Block
# ========================
def residual_block(x, units):
    shortcut = x
    x = Dense(units, kernel_regularizer=Config.REGULARIZATION)(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Add()([shortcut, x])  # Residual connection
    return x


# ========================
# Model Architecture
# ========================
def create_model(input_shape):
    inputs = Input(shape=(input_shape,))

    x = Dense(256, kernel_regularizer=Config.REGULARIZATION)(inputs)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Add residual blocks
    x = residual_block(x, 128)
    x = residual_block(x, 128)

    x = Dense(64, kernel_regularizer=Config.REGULARIZATION)(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    outputs = Dense(len(Config.TARGET_COLS))(x)

    model = Model(inputs, outputs)
    return model


# ========================
# Training Pipeline
# ========================
def main():
    # Load data
    project_root = Path(__file__).resolve().parents[3]
    data_path = project_root / "Training_set" / "New_Training_set.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")

    train_df = pd.read_csv(data_path)
    X = train_df[Config.FEATURE_COLS].values.astype('float32')
    y = train_df[Config.TARGET_COLS].values.astype('float32')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE, random_state=Config.SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=Config.VAL_SIZE,
                                                      random_state=Config.SEED)

    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler.pkl')

    # Model setup
    tf.keras.backend.clear_session()
    model = create_model(X_train_scaled.shape[1])

    # Use Huber loss and gradient clipping
    optimizer = Adam(learning_rate=Config.INITIAL_LR, clipvalue=Config.CLIP_VALUE)
    model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])

    # Callbacks
    callbacks = [
        EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(patience=Config.REDUCE_LR_PATIENCE, factor=0.5, min_lr=1e-6),
        ModelCheckpoint(Config.CHECKPOINT_PATH, save_best_only=True)
    ]

    # Training
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate model
    test_pred = model.predict(X_test_scaled)
    r2_scores = [r2_score(y_test[:, i], test_pred[:, i]) for i in range(len(Config.TARGET_COLS))]
    mae_scores = [mean_absolute_error(y_test[:, i], test_pred[:, i]) for i in range(len(Config.TARGET_COLS))]

    # Display Results
    print("\n=== Final Performance ===")
    print(f"Overall R2 Score: {np.mean(r2_scores) * 100:.2f}%")
    print(f"Average MAE: {np.mean(mae_scores):.3f} dB")
    for i, target in enumerate(Config.TARGET_COLS):
        print(f"{target}:")
        print(f"R2 Score: {r2_scores[i] * 100:.2f}%")
        print(f"MAE: {mae_scores[i]:.3f} dB")

    # Plot performance
    for i, target in enumerate(Config.TARGET_COLS):
        plt.figure(figsize=(10, 4))
        plt.scatter(y_test[:, i], test_pred[:, i], alpha=0.5)
        plt.plot([min(y_test[:, i]), max(y_test[:, i])], [min(y_test[:, i]), max(y_test[:, i])], color='red')
        plt.title(f"{target} - Predicted vs Actual")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.show()


if __name__ == "__main__":
    main()