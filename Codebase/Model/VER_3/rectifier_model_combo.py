import os
import warnings
import re
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
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import keras_tuner as kt
import joblib
import mlflow
from scipy import stats
from pathlib import Path


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
    EPOCHS = 50
    BATCH_SIZE = 128
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    REGULARIZATION = l1_l2(l1=1e-5, l2=1e-4)
    FEATURE_COLS = ["l_s", "l_2", "l_1", "s_2", "s_1", "w_s", "w_2", "w_1", "freq"]
    TARGET_COLS = ["dB(S(1,1))", "dB(S(2,1))"]
    CHECKPOINT_PATH = "best_model.h5"


# Reproducibility
np.random.seed(Config.SEED)
tf.random.set_seed(Config.SEED)


# =======================
# Data Loading Fix
# =======================
def parse_frequency(freq_str):
    """Convert a frequency string with a unit (MHz or GHz) into a numeric value in MHz."""
    if not isinstance(freq_str, str):
        try:
            return float(freq_str)
        except Exception:
            return np.nan

    freq_str = freq_str.replace("\xa0", " ").strip().strip('"').strip("'").lower()
    numeric_part = re.sub(r'[^0-9\.]', '', freq_str)

    if numeric_part == "":
        return np.nan

    try:
        number = float(numeric_part)
    except ValueError:
        return np.nan

    if "ghz" in freq_str:
        return number * 1000
    elif "mhz" in freq_str:
        return number
    return number  # Assume it's already in MHz if no unit found


def load_data(path, is_generated=False):
    """Load CSV with proper column handling."""
    if is_generated:
        df = pd.read_csv(path, delimiter=",")
        if "ID" in df.columns:
            df = df.drop(columns=["ID"])
    else:
        df = pd.read_csv(path, skiprows=1, header=None, names=Config.FEATURE_COLS + Config.TARGET_COLS)

    df["freq"] = df["freq"].apply(parse_frequency)
    df["freq"] = pd.to_numeric(df["freq"], errors='coerce')

    if not is_generated:
        numeric_cols = Config.FEATURE_COLS + Config.TARGET_COLS
        z = np.abs(stats.zscore(df[numeric_cols]))
        df = df[(z < 3).all(axis=1)]

    return df


# ========================
# Visualization Functions
# ========================
def plot_individual_performance(y_true, y_pred, target_name):
    """Generate separate analysis for each target parameter."""
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
    plt.savefig(f"{target_name}_performance.png")
    plt.close()


def plot_correlation_heatmap(df):
    """Plots a heatmap showing feature correlation."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.savefig("correlation_heatmap.png")
    plt.close()


# ====================
# Hyperparameter-Optimized Model
# ====================
def build_model(hp):
    inputs = Input(shape=(len(Config.FEATURE_COLS),))

    activation_1 = hp.Choice('activation_1', values=['relu', 'tanh'])
    activation_2 = hp.Choice('activation_2', values=['relu', 'tanh'])
    activation_3 = hp.Choice('activation_2', values=['relu', 'tanh'])

    x = Dense(256, activation=activation_1, kernel_regularizer=Config.REGULARIZATION)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(128, activation=activation_2, kernel_regularizer=Config.REGULARIZATION)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(64, activation=activation_3, kernel_regularizer=Config.REGULARIZATION)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    outputs = Dense(len(Config.TARGET_COLS))(x)
    model = Model(inputs, outputs)

    learning_rate = hp.Float('learning_rate', min_value=0.0001, max_value=0.01, sampling='LOG')

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model


# ===================
# Training Pipeline
# ===================
def main():
    # Load Data
    # Find the project root dynamically
    project_root = Path(__file__).resolve().parents[3]  # Go up 4 levels from script directory
    # Construct the correct data path
    data_path = project_root / "Training_set" / "New_Training_set.csv"
    # Check if the file exists before loading
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")

    # Convert Path object to string for compatibility
    train_df = load_data(str(data_path), is_generated=False)

    # Prepare data
    X = train_df[Config.FEATURE_COLS].values.astype('float32')
    y = train_df[Config.TARGET_COLS].values.astype('float32')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE, random_state=Config.SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=Config.VAL_SIZE, random_state=Config.SEED)

    # Feature scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler.pkl')

    # MLflow setup
    mlflow.set_tracking_uri("mlruns")
    mlflow.tensorflow.autolog()

    with mlflow.start_run():
        # Hyperparameter Tuning
        tuner = kt.RandomSearch(
            build_model,
            objective='val_mae',
            max_trials=10,
            directory='tuner_results',
            project_name='activation_function_tuning'
        )

        tuner.search(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val),
                     epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, verbose=1)

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.get_best_models(num_models=1)[0]

        # Evaluate
        test_pred = best_model.predict(X_test_scaled)
        r2_scores = [r2_score(y_test[:, i], test_pred[:, i]) for i in range(len(Config.TARGET_COLS))]
        mae_scores = [mean_absolute_error(y_test[:, i], test_pred[:, i]) for i in range(len(Config.TARGET_COLS))]

        print(f"Best Activation: {best_hps.get('activation_1')} & {best_hps.get('activation_2')}")
        print(f"Best Learning Rate: {best_hps.get('learning_rate'):.6f}")

        for i, target in enumerate(Config.TARGET_COLS):
            plot_individual_performance(y_test[:, i], test_pred[:, i], target)

        best_model.save(Config.CHECKPOINT_PATH)

        # === After model.fit() and evaluation ===

        # Extract model architecture details dynamically
        hidden_layers = [layer for layer in best_model.layers if isinstance(layer, Dense)][:-1]  # Exclude output layer
        num_hidden_layers = len(hidden_layers)
        neurons_per_layer = [layer.units for layer in hidden_layers]
        activations = [layer.activation.__name__ for layer in hidden_layers]

        # Print hyperparameters and training info
        print("\n=== Model Hyperparameters and Training Info ===")
        print(f"Activation function (hidden layers): {activations}")
        print(f"Number of hidden layers: {num_hidden_layers}")
        print(f"Neurons in each hidden layer: {neurons_per_layer}")

        # Get initial learning rate
        initial_lr = best_hps.get('learning_rate')
        print(f"Initial learning rate: {initial_lr:.6f}")

        # Obtain the current (final) learning rate from the optimizer (useful if ReduceLROnPlateau modified it)
        final_lr = tf.keras.backend.get_value(best_model.optimizer.lr)
        print(f"Best (final) learning rate: {final_lr:.6f}")

        # Number of epochs (since history is missing, estimate based on early stopping if used)
        epochs_run = Config.EPOCHS  # Assuming full run; adjust if early stopping applies
        print(f"Number of epochs run: {epochs_run}")

        # Print the optimizer (adaptive learning function)
        print(f"Adaptive learning function (optimizer): {type(best_model.optimizer).__name__}")


if __name__ == "__main__":
    main()
