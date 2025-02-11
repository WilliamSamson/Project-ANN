import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
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

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")


# ========================
# Configuration Parameters
# ========================
class Config:
    SEED = 42
    TEST_SIZE = 0.20
    VAL_SIZE = 0.10
    EPOCHS = 50  # Reduced for speed
    BATCH_SIZE = 128  # Larger batch size for faster training
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    REGULARIZATION = l1_l2(l1=1e-5, l2=1e-4)
    TARGET_NAMES = ["dB(S(1,1))", "dB(S(2,1))"]
    CHECKPOINT_PATH = "best_model.h5"


# Reproducibility
np.random.seed(Config.SEED)
tf.random.set_seed(Config.SEED)


# =======================
# Data Loading
# =======================
def parse_frequency(freq_str):
    """Convert frequency string to MHz."""
    if isinstance(freq_str, str):
        freq_str = freq_str.strip()
        if "MHz" in freq_str:
            return float(freq_str.replace("MHz", ""))
        elif "GHz" in freq_str:
            return float(freq_str.replace("GHz", "")) * 1000
    return float(freq_str)


def load_data(path):
    """Load and preprocess data."""
    df = pd.read_csv(path)
    if "freq" in df.columns:
        df["freq"] = df["freq"].apply(parse_frequency)
    numeric_cols = df.select_dtypes(include=np.number).columns
    z = np.abs(stats.zscore(df[numeric_cols]))
    return df[(z < 3).all(axis=1)]


# ======================
# Enhanced Visualization
# ======================
def plot_individual_analysis(y_true, y_pred, target_name):
    """Generate separate plots for each target."""
    plt.figure(figsize=(12, 5))

    # Prediction vs Actual
    plt.subplot(1, 2, 1)
    sns.regplot(x=y_true, y=y_pred, scatter_kws={'alpha': 0.3})
    plt.title(f'{target_name} - Actual vs Predicted')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    # Error distribution
    plt.subplot(1, 2, 2)
    errors = y_true - y_pred
    sns.histplot(errors, kde=True)
    plt.title(f'{target_name} Error Distribution')
    plt.tight_layout()
    plt.savefig(f"{target_name}_analysis.png")
    plt.close()


# ====================
# Advanced Model
# ====================
def create_model(input_shape):
    """Optimized neural network architecture."""
    inputs = Input(shape=(input_shape,))

    x = Dense(256, activation='relu', kernel_regularizer=Config.REGULARIZATION)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(128, activation='relu', kernel_regularizer=Config.REGULARIZATION)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    outputs = Dense(len(Config.TARGET_NAMES))(x)
    return Model(inputs, outputs)


# ===================
# Training Pipeline
# ===================
def main():
    # Load data with corrected paths
    train_df = load_data('/home/kayode-olalere/PycharmProjects/Project ANN/Training_set/New_Training_set.csv')
    generated_df = load_data(
        '/home/kayode-olalere/PycharmProjects/Project ANN/Codebase/Data_Gen/generated_input_dataset.csv')

    # Prepare data
    features = ["l_s", "l_2", "l_1", "s_2", "s_1", "w_s", "w_2", "w_1", "freq"]
    X = train_df[features].values.astype('float32')
    y = train_df[Config.TARGET_NAMES].values.astype('float32')

    # Train/val/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE, random_state=Config.SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=Config.VAL_SIZE,
                                                      random_state=Config.SEED)

    # Feature scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'robust_scaler.pkl')

    # MLflow tracking
    mlflow.set_tracking_uri("mlruns")
    mlflow.tensorflow.autolog()

    with mlflow.start_run():
        # Build and compile model
        model = create_model(X_train_scaled.shape[1])
        model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])

        # Callbacks
        callbacks = [
            EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE, restore_best_weights=True),
            ReduceLROnPlateau(patience=Config.REDUCE_LR_PATIENCE, factor=0.5),
            ModelCheckpoint(Config.CHECKPOINT_PATH, save_best_only=True)
        ]

        # Train with optimized batches
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )

        # Generate predictions
        test_pred = model.predict(X_test_scaled)

        # Calculate metrics
        r2_scores = [r2_score(y_test[:, i], test_pred[:, i]) for i in range(len(Config.TARGET_NAMES))]
        overall_r2 = np.mean(r2_scores)

        # Layman-friendly output
        print("\n=== Final Results ===")
        print(f"Model Accuracy: {overall_r2 * 100:.1f}%")
        print(f"Average Error: {np.mean(mean_absolute_error(y_test, test_pred, multioutput='raw_values')):.4f} dB")

        # Individual target analysis
        for i, target in enumerate(Config.TARGET_NAMES):
            plot_individual_analysis(y_test[:, i], test_pred[:, i], target)

            # Target-specific metrics
            print(f"\n{target} Performance:")
            print(f"- R²: {r2_scores[i]:.3f} (Accuracy: {r2_scores[i] * 100:.1f}%)")
            print(f"- MAE: {mean_absolute_error(y_test[:, i], test_pred[:, i]):.4f} dB")

        # Generate frequency analysis
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=train_df['freq'], y=train_df[Config.TARGET_NAMES[0]], errorbar=None)
        plt.title("Frequency vs S-Parameter Performance")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("dB Magnitude")
        plt.savefig("frequency_analysis.png")
        plt.close()

        # Save predictions
        gen_inputs = generated_df[features].values.astype('float32')
        gen_pred = model.predict(scaler.transform(gen_inputs))
        generated_df[Config.TARGET_NAMES] = gen_pred
        generated_df.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()