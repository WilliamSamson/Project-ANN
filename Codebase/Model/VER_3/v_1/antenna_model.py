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
    import re
    # If it's not a string, try converting directly.
    if not isinstance(freq_str, str):
        try:
            return float(freq_str)
        except Exception:
            return np.nan

    # Remove non-breaking spaces and any surrounding quotes.
    freq_str = freq_str.replace("\xa0", " ").strip().strip('"').strip("'")
    lower_str = freq_str.lower()

    # Try extracting the numeric part by removing all non-digit and non-dot characters.
    numeric_part = re.sub(r'[^0-9\.]', '', freq_str)
    if numeric_part == "":
        # Fallback: if nothing remains, try splitting on space.
        parts = freq_str.split()
        if parts:
            try:
                number = float(parts[0])
            except ValueError:
                return np.nan
        else:
            return np.nan
    else:
        try:
            number = float(numeric_part)
        except ValueError:
            return np.nan

    if "ghz" in lower_str:
        return number * 1000  # Convert GHz to MHz.
    elif "mhz" in lower_str:
        return number
    else:
        # If no unit is found, assume the number is already in MHz.
        return number

def load_data(path, is_generated=False):
    """Load CSV with proper column handling."""
    if is_generated:
        # For generated data, use the header provided in the file.
        df = pd.read_csv(path, delimiter=",")
        # Drop the 'ID' column if it's not needed.
        if "ID" in df.columns:
            df = df.drop(columns=["ID"])
        # For generated data, our expected columns are:
        # ["l_s", "l_2", "l_1", "w_s", "w_2", "w_1", "freq"]
        # (Note: It is missing s_2, s_1, and target columns.)
    else:
        # For training data, skip the metadata row and assign proper names.
        df = pd.read_csv(
            path,
            skiprows=1,
            header=None,
            names=Config.FEATURE_COLS + Config.TARGET_COLS
        )

    # Process frequency column: regardless of file, process "freq" column.
    df["freq"] = df["freq"].apply(parse_frequency)
    df["freq"] = pd.to_numeric(df["freq"], errors='coerce')

    # If this file is training data, remove outliers using z-score.
    # For generated data, you might not need to perform outlier removal.
    if not is_generated:
        numeric_cols = Config.FEATURE_COLS + Config.TARGET_COLS
        z = np.abs(stats.zscore(df[numeric_cols]))
        df = df[(z < 3).all(axis=1)]

    return df

# ======================
# Enhanced Visualization
# ======================
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


# ====================
# Model Architecture
# ====================
def create_model(input_shape):
    inputs = Input(shape=(input_shape,))

    x = Dense(256, activation='relu', kernel_regularizer=Config.REGULARIZATION)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(128, activation='relu', kernel_regularizer=Config.REGULARIZATION)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    outputs = Dense(len(Config.TARGET_COLS))(x)
    return Model(inputs, outputs)

# ========================
# Correlation Heatmap Plot
# ========================
def plot_correlation_heatmap(df, save_path="correlation_heatmap.png"):
    """Plots a heatmap showing the correlation between features and target variables."""
    plt.figure(figsize=(12, 8))
    corr_matrix = df.corr()

    # Create heatmap
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5
    )

    plt.title("Feature Correlation Heatmap")
    plt.savefig(save_path)
    plt.close()
    print(f"Correlation heatmap saved as {save_path}")

# ===================
# Training Pipeline
# ===================
def main():
    # Load data with corrected paths
    train_df = load_data('/home/kayode-olalere/PycharmProjects/Project ANN/Training_set/New_Training_set.csv',is_generated=False)
    generated_df = load_data(
        '/home/kayode-olalere/PycharmProjects/Project ANN/Codebase/Data_Gen/generated_input_dataset.csv',is_generated=True)

    # Prepare data
    X = train_df[Config.FEATURE_COLS].values.astype('float32')
    y = train_df[Config.TARGET_COLS].values.astype('float32')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE, random_state=Config.SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=Config.VAL_SIZE,
                                                      random_state=Config.SEED)

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
        # Build and compile model
        model = create_model(X_train_scaled.shape[1])
        model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])

        # Callbacks
        callbacks = [
            EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE, restore_best_weights=True),
            ReduceLROnPlateau(patience=Config.REDUCE_LR_PATIENCE, factor=0.5),
            ModelCheckpoint(Config.CHECKPOINT_PATH, save_best_only=True)
        ]

        # Train model
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate
        test_pred = model.predict(X_test_scaled)

        # Calculate metrics
        r2_scores = [r2_score(y_test[:, i], test_pred[:, i]) for i in range(2)]
        mae_scores = [mean_absolute_error(y_test[:, i], test_pred[:, i]) for i in range(2)]

        # Layman-friendly output
        print("\n=== Final Performance ===")
        print(f"Overall Accuracy: {np.mean(r2_scores) * 100:.1f}%")
        print(f"Average Error: {np.mean(mae_scores):.2f} dB")
        print("\nParameter-wise Performance:")
        for i, target in enumerate(Config.TARGET_COLS):
            print(f"{target}:")
            print(f"- Accuracy: {r2_scores[i] * 100:.1f}%")
            print(f"- Average Error: {mae_scores[i]:.2f} dB")
            plot_individual_performance(y_test[:, i], test_pred[:, i], target)

        # Frequency analysis
        plt.figure(figsize=(10, 6))
        for target in Config.TARGET_COLS:
            sns.lineplot(x=train_df['freq'], y=train_df[target], label=target, errorbar=None)
        plt.title("Frequency Response")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("dB Magnitude")
        plt.legend()
        plt.savefig("frequency_response.png")
        plt.close()

        # Call the function
        plot_correlation_heatmap(train_df)

        # Generate predictions
        gen_inputs = generated_df[Config.FEATURE_COLS].values.astype('float32')
        gen_pred = model.predict(scaler.transform(gen_inputs))
        generated_df[Config.TARGET_COLS] = gen_pred
        generated_df.to_csv("antenna_predictions.csv", index=False)

        # === After model.fit() and evaluation ===

        # Extract model architecture details dynamically
    hidden_layers = [layer for layer in model.layers if isinstance(layer, Dense)][:-1]  # Exclude output layer
    num_hidden_layers = len(hidden_layers)
    neurons_per_layer = [layer.units for layer in hidden_layers]
    activations = [layer.activation.__name__ for layer in hidden_layers]

    # Print hyperparameters and training info
    print("\n=== Model Hyperparameters and Training Info ===")
    print(f"Activation function (hidden layers): {activations}")
    print(f"Number of hidden layers: {num_hidden_layers}")
    print(f"Neurons in each hidden layer: {neurons_per_layer}")

    # Get initial learning rate
    initial_lr = model.optimizer.learning_rate.numpy()
    print(f"Initial learning rate: {initial_lr:.6f}")

    # Obtain the current (final) learning rate from the optimizer (useful if ReduceLROnPlateau modified it)
    final_lr = tf.keras.backend.get_value(model.optimizer.lr)
    print(f"Best (final) learning rate: {final_lr:.6f}")

    # The number of epochs run is the length of the training history.
    epochs_run = len(history.history['loss'])
    print(f"Number of epochs run: {epochs_run}")

    # Print the optimizer (adaptive learning function)
    print(f"Adaptive learning function (optimizer): {type(model.optimizer).__name__}")


if __name__ == "__main__":
    main()
