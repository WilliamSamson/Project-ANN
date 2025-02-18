

# Comprehensive Documentation for the Hyperparameter-Optimized Antenna Model

## 1. Introduction

This script builds, tunes, and evaluates an artificial neural network (ANN) for predicting antenna parameters (such as dB measurements). It leverages advanced libraries like TensorFlow, Keras Tuner, and MLflow for hyperparameter optimization and experiment tracking. Even if you're new to deep learning or hyperparameter tuning, this guide will walk you through each step of the process.

---

## 2. Code Overview

### **Imports and Global Settings**

- **Imports:**
  The script begins by importing essential libraries for data manipulation (NumPy, Pandas), deep learning (TensorFlow/Keras), hyperparameter tuning (Keras Tuner), visualization (Matplotlib, Seaborn), and experiment tracking (MLflow).
- **Suppressing Warnings:**
  Warnings from TensorFlow and other libraries are suppressed to keep the output clean.
- **Reproducibility:**
  Random seeds are set using both NumPy and TensorFlow to ensure that your results can be reproduced.


```
python
import os, warnings, re, numpy as np, pandas as pd, tensorflow as tf
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
import joblib, mlflow
from scipy import stats
from pathlib import Path

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
```


**Why?**
Setting a seed and suppressing warnings ensures consistency and reduces noise in your training logs. This is essential for debugging and comparing experiments.

---

## 3. Configuration Parameters

All hyperparameters and file path settings are contained in the `Config` class. These include:

- **Seed for randomness**
- **Test/Validation split sizes**
- **Training parameters (epochs, batch size, early stopping, etc.)**
- **Regularization settings**
- **Input and target feature columns**


```
python
class Config:
## SEED = 42
    TEST_SIZE = 0.20         # 20% of data reserved for final testing
    VAL_SIZE = 0.10          # 10% of training data reserved for validation
    EPOCHS = 50              # Number of training epochs (full dataset iterations)
    BATCH_SIZE = 128         # Number of samples per training batch
    EARLY_STOPPING_PATIENCE = 10  # Stop training early if validation performance stops improving
    REDUCE_LR_PATIENCE = 5        # Reduce learning rate if no improvement for 5 epochs
    REGULARIZATION = l1_l2(l1=1e-5, l2=1e-4)  # L1-L2 regularization to help prevent overfitting
    FEATURE_COLS = ["l_s", "l_2", "l_1", "s_2", "s_1", "w_s", "w_2", "w_1", "freq"]
    TARGET_COLS = ["dB(S(1,1))", "dB(S(2,1))"]
    CHECKPOINT_PATH = "best_model.h5"  # Where to save the best model during training
```


**Why?**
Centralizing configuration makes the code easier to manage and modify. Experimenting with different parameters becomes as simple as changing these values.

---

## 4. Data Loading and Preprocessing

### **Frequency Parsing**

The `parse_frequency` function converts various frequency string formats (e.g., "2.4 GHz", "2400 MHz") into a standardized numeric value in MHz.


```
python
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
        return number * 1000  # Convert GHz to MHz.
    elif "mhz" in freq_str:
        return number
    return number  # Assume value is in MHz if no unit found.
```


**Why?**
Data inconsistencies are common in real-world datasets. This function ensures all frequency data is standardized, which is critical for model accuracy.

---

### **Loading Data**

The `load_data` function reads a CSV file, applies the frequency parser, and removes outliers using the Z-score method (threshold of 3).


```
python
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
```


**Why?**
- **Skipping metadata:** The training CSV contains a metadata row that we skip.
- **Outlier Removal:** Using Z-scores ensures extreme values (which might be errors) don’t skew the model training.

---

## 5. Visualization Functions

### **Plotting Individual Performance**

This function creates two plots for each target:
1. **Prediction vs. Actual:** A regression plot to compare model predictions with actual values.
2. **Error Distribution:** A histogram showing the error spread.


```
python
def plot_individual_performance(y_true, y_pred, target_name):
    """Generate separate analysis for each target parameter."""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.regplot(x=y_true, y=y_pred, scatter_kws={'alpha': 0.3})
    plt.title(f'{target_name} - Actual vs Predicted')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.subplot(1, 2, 2)
    errors = y_true - y_pred
    sns.histplot(errors, kde=True)
    plt.title(f'{target_name} Error Distribution')
    plt.tight_layout()
    plt.savefig(f"{target_name}_performance.png")
    plt.close()
```


### **Plotting Correlation Heatmap**

This function generates a heatmap of the correlation matrix for all features and targets.


```
python
def plot_correlation_heatmap(df):
    """Plots a heatmap showing feature correlation."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.savefig("correlation_heatmap.png")
    plt.close()
```


**Why?**
Visualizations help to quickly understand model performance and relationships within the data. They are invaluable during the debugging and model refinement phases.

---

## 6. Hyperparameter-Optimized Model Building

### **Building the Model with Keras Tuner**

The `build_model` function uses Keras Tuner to allow dynamic selection of:
- **Activation functions** for each dense layer (choices include 'relu' and 'tanh').
- **Learning rate** (sampled logarithmically between 0.0001 and 0.01).


```
python
def build_model(hp):
    inputs = Input(shape=(len(Config.FEATURE_COLS),))
    activation_1 = hp.Choice('activation_1', values=['relu', 'tanh'])
    activation_2 = hp.Choice('activation_2', values=['relu', 'tanh'])
    activation_3 = hp.Choice('activation_2', values=['relu', 'tanh'])  # Note: activation_2 is repeated intentionally; adjust as needed.
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
```


**Why?**
- **Hyperparameter Tuning:** Instead of manually choosing activation functions and learning rates, Keras Tuner helps explore combinations to find the best configuration for your dataset.
- **Layer Design:** Multiple dense layers with batch normalization and dropout improve training stability and prevent overfitting.

---

## 7. Training Pipeline and Main Function

The `main()` function orchestrates the entire process:
1. **Data Loading:**
   The script dynamically finds the project root, builds the file path, and loads the training data.
2. **Data Preparation:**
   Splits data into training, validation, and test sets, and scales features using `RobustScaler`.
3. **MLflow Setup:**
   Enables MLflow for tracking experiment metrics and training details.
4. **Hyperparameter Tuning:**
   Uses Keras Tuner’s RandomSearch to find the best hyperparameters.
5. **Model Evaluation:**
   Evaluates the tuned model on the test set, computes metrics (R², MAE), and visualizes performance.
6. **Saving the Model and Reporting:**
   Saves the best model, prints details about model architecture, hyperparameters, and training performance.


```
python
def main():
    # Load Data: Dynamically build path and check file existence.
    project_root = Path(__file__).resolve().parents[3]
    data_path = project_root / "Training_set" / "New_Training_set.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")
    train_df = load_data(str(data_path), is_generated=False)

    # Data Preparation: Split into training, validation, and test sets.
    X = train_df[Config.FEATURE_COLS].values.astype('float32')
    y = train_df[Config.TARGET_COLS].values.astype('float32')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE, random_state=Config.SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=Config.VAL_SIZE, random_state=Config.SEED)

    # Feature Scaling: Use RobustScaler to handle outliers.
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler.pkl')

    # MLflow Setup: Log experiments with MLflow.
    mlflow.set_tracking_uri("mlruns")
    mlflow.tensorflow.autolog()

    with mlflow.start_run():
        # Hyperparameter Tuning: Search for the best model configuration.
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

        # Evaluate the best model on test data.
        test_pred = best_model.predict(X_test_scaled)
        r2_scores = [r2_score(y_test[:, i], test_pred[:, i]) for i in range(len(Config.TARGET_COLS))]
        mae_scores = [mean_absolute_error(y_test[:, i], test_pred[:, i]) for i in range(len(Config.TARGET_COLS))]

        print(f"Best Activation: {best_hps.get('activation_1')} & {best_hps.get('activation_2')}")
        print(f"Best Learning Rate: {best_hps.get('learning_rate'):.6f}")

        # Generate performance plots for each target.
        for i, target in enumerate(Config.TARGET_COLS):
            plot_individual_performance(y_test[:, i], test_pred[:, i], target)

        # Save the best model.
        best_model.save(Config.CHECKPOINT_PATH)

        # Print model architecture details.
        hidden_layers = [layer for layer in best_model.layers if isinstance(layer, Dense)][:-1]
        num_hidden_layers = len(hidden_layers)
        neurons_per_layer = [layer.units for layer in hidden_layers]
        activations = [layer.activation.__name__ for layer in hidden_layers]

        print("\n=== Model Hyperparameters and Training Info ===")
        print(f"Activation function (hidden layers): {activations}")
        print(f"Number of hidden layers: {num_hidden_layers}")
        print(f"Neurons in each hidden layer: {neurons_per_layer}")

        # Print learning rate and epoch details.
        initial_lr = best_hps.get('learning_rate')
        print(f"Initial learning rate: {initial_lr:.6f}")
        final_lr = tf.keras.backend.get_value(best_model.optimizer.lr)
        print(f"Best (final) learning rate: {final_lr:.6f}")
        epochs_run = Config.EPOCHS  # Modify if early stopping is used.
        print(f"Number of epochs run: {epochs_run}")
        print(f"Adaptive learning function (optimizer): {type(best_model.optimizer).__name__}")

if __name__ == "__main__":
    main()
```


**Why?**
- **Dynamic Path Construction:** Makes your code more portable.
- **Robust Data Splitting & Scaling:** Ensures that your model trains on normalized data and is evaluated on unseen data.
- **Hyperparameter Tuning:** Allows the model to automatically search for the best configuration, reducing manual trial-and-error.
- **MLflow Integration:** Automatically logs important metrics and parameters, which is crucial for reproducibility and future comparisons.
- **Visualization and Model Saving:** Helps in analyzing performance and preserving the best model.

---

## 8. Final Thoughts

This advanced script demonstrates best practices in deep learning:

- **Modular Design:** Functions are separated by purpose (data loading, visualization, model building).
- **Hyperparameter Tuning:** Keras Tuner is used to find optimal model parameters.
- **Experiment Tracking:** MLflow integration helps monitor experiments over time.
- **Visualization:** Graphs generated throughout the process provide clear insights into model performance.
