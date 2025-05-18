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
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import mlflow.tensorflow
import seaborn as sns


# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

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

IMAGE_SAVE_DIR = "Graphs"
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)  # Create directory if it doesn't exist


def parse_frequency(freq_str):
    """Convert a frequency string with a unit (MHz or GHz) into a numeric value in MHz."""

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
    plt.savefig(os.path.join(IMAGE_SAVE_DIR, f"{target_name}_performance.png"))
    plt.close()


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

def plot_correlation_heatmap(df ):
    """Plots a heatmap showing the correlation between features and target variables."""
    plt.figure(figsize=(12, 8))
    corr_matrix = df.corr()

    # Create heatmap
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5
    )

    plt.title("Feature Correlation Heatmap")
    plt.savefig(os.path.join(IMAGE_SAVE_DIR,"correlation_heatmap.png"))
    plt.close()
    print(f"Correlation heatmap saved")




def main():
    project_root = Path(__file__).resolve().parents[3]
    data_path = project_root / "Training_set" / "training_output_fixed.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")

    train_df = load_data(str(data_path), is_generated=False)

    X = train_df[Config.FEATURE_COLS].values.astype('float32')
    y = train_df[Config.TARGET_COLS].values.astype('float32')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE, random_state=Config.SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=Config.VAL_SIZE, random_state=Config.SEED)

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler.pkl')

    def detect_out_of_bounds(X_train, X_input):
        bounds_min = np.min(X_train, axis=0)
        bounds_max = np.max(X_train, axis=0)
        out_of_bounds_mask = (X_input < bounds_min) | (X_input > bounds_max)
        return out_of_bounds_mask.any(axis=1)

    extrapolation_flags = detect_out_of_bounds(X_train_scaled, X_test_scaled)
    num_extrapolated = np.sum(extrapolation_flags)
    if num_extrapolated > 0:
        print(f"[Warning] {num_extrapolated} samples in test set are outside training distribution bounds.")

    mlflow.set_tracking_uri("mlruns")
    mlflow.tensorflow.autolog()

    with mlflow.start_run():
        model = create_model(X_train_scaled.shape[1])
        model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])

        callbacks = [
            EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE, restore_best_weights=True),
            ReduceLROnPlateau(patience=Config.REDUCE_LR_PATIENCE, factor=0.5),
            ModelCheckpoint(Config.CHECKPOINT_PATH, save_best_only=True)
        ]

        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )

        test_pred = model.predict(X_test_scaled)

        if num_extrapolated > 0:
            poly_model = make_pipeline(PolynomialFeatures(degree=3), Ridge(alpha=1.0))
            poly_model.fit(X_train_scaled, y_train)

            X_extra = X_test_scaled[extrapolation_flags]
            y_extra_true = y_test[extrapolation_flags]
            y_extra_pred = poly_model.predict(X_extra)

            r2_extra = [r2_score(y_extra_true[:, i], y_extra_pred[:, i]) for i in range(2)]
            print("\n=== Polynomial Fallback Model (Extrapolation Region) ===")
            for i, target in enumerate(Config.TARGET_COLS):
                print(f"{target}:")
                print(f"- Fallback R²: {r2_extra[i]:.2f}")

            test_pred[extrapolation_flags] = y_extra_pred

            extrapolated_df = pd.DataFrame(X_test[extrapolation_flags], columns=Config.FEATURE_COLS)
            extrapolated_df.to_csv("extrapolated_inputs.csv", index=False)

        r2_scores = [r2_score(y_test[:, i], test_pred[:, i]) for i in range(2)]
        mae_scores = [mean_absolute_error(y_test[:, i], test_pred[:, i]) for i in range(2)]

        plot_correlation_heatmap(train_df)

        print("\n=== Final Performance ===")
        print(f"Overall Accuracy: {np.mean(r2_scores) * 100:.1f}%")
        print(f"Average Error: {np.mean(mae_scores):.2f} dB")
        print("\nParameter-wise Performance:")
        for i, target in enumerate(Config.TARGET_COLS):
            print(f"{target}:")
            print(f"- Accuracy: {r2_scores[i] * 100:.1f}%")
            print(f"- Average Error: {mae_scores[i]:.2f} dB")
            plot_individual_performance(y_test[:, i], test_pred[:, i], target)

        plt.figure(figsize=(10, 6))
        for target in Config.TARGET_COLS:
            sns.lineplot(x=train_df['freq'], y=train_df[target], label=target, errorbar=None)
        plt.title("Frequency Response")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("dB Magnitude")
        plt.legend()
        plt.savefig(os.path.join(IMAGE_SAVE_DIR, "frequency_response.png"))
        plt.close()

        hidden_layers = [layer for layer in model.layers if isinstance(layer, Dense)][:-1]
        num_hidden_layers = len(hidden_layers)
        neurons_per_layer = [layer.units for layer in hidden_layers]
        activations = [layer.activation.__name__ for layer in hidden_layers]

        print("\n=== Model Hyperparameters and Training Info ===")
        print(f"Activation function (hidden layers): {activations}")
        print(f"Number of hidden layers: {num_hidden_layers}")
        print(f"Neurons in each hidden layer: {neurons_per_layer}")

        initial_lr = model.optimizer.learning_rate.numpy()
        print(f"Initial learning rate: {initial_lr:.6f}")

        if isinstance(model.optimizer.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
            final_lr = model.optimizer.learning_rate(tf.keras.backend.get_value(model.optimizer.iterations)).numpy()
        else:
            final_lr = tf.keras.backend.get_value(model.optimizer.learning_rate)

        print(f"Best (final) learning rate: {final_lr:.6f}")
        print(f"Number of epochs run: {len(history.history['loss'])}")
        print(f"Adaptive learning function (optimizer): {type(model.optimizer).__name__}")

if __name__ == "__main__":
    main()
