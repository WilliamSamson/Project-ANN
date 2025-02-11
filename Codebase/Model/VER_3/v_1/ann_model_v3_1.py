import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import mlflow
import joblib

# Suppress unnecessary TensorFlow and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", message=".*DejaVu Sans.*")


# ========================
# Configuration Parameters
# ========================
class Config:
    SEED = 42
    TEST_SIZE = 0.20  # 20% hold-out test set
    VAL_SIZE = 0.10  # 10% of training data as validation
    EPOCHS = 100  # Reduced epochs for quicker training
    BATCH_SIZE = 32
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    REGULARIZATION = l1_l2(l1=1e-5, l2=1e-4)
    TARGET_NAMES = ["dB(S(1,1))", "dB(S(2,1))"]
    EXPLORATION_PLOTS = True
    CHECKPOINT_PATH = "best_light_model.h5"
    RUN_SHAP = False  # SHAP analysis can be heavy—disable on low-power machines


# For reproducibility
np.random.seed(Config.SEED)
tf.random.set_seed(Config.SEED)


# =======================
# Data Loading and Helpers
# =======================
def parse_frequency(freq_str):
    """
    Convert frequency string to a float in MHz.
    e.g., '800.0 MHz' -> 800.0, '1.000 GHz' -> 1000.0.
    """
    if isinstance(freq_str, str):
        freq_str = freq_str.strip()
        if "MHz" in freq_str:
            return float(freq_str.replace("MHz", "").strip())
        elif "GHz" in freq_str:
            return float(freq_str.replace("GHz", "").strip()) * 1000
    return float(freq_str)


def load_data(path):
    """Load CSV data, convert frequency strings, and remove numeric outliers."""
    df = pd.read_csv(path)
    if "freq" in df.columns:
        df["freq"] = df["freq"].apply(parse_frequency)

    # Outlier removal using Z-score (keep only rows where all numeric z-scores < 3)
    numeric_cols = df.select_dtypes(include=np.number).columns
    z = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z < 3).all(axis=1)]
    return df


# ===========================
# Light-Weight Model Definition
# ===========================
def create_light_model(input_shape):
    """
    Create a simplified, lightweight neural network model.
    This design trades a bit of complexity for much faster training on lower-end PCs.
    """
    inputs = Input(shape=(input_shape,))

    # First hidden layer
    x = Dense(128, kernel_regularizer=Config.REGULARIZATION)(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = Dropout(0.2)(x)

    # Second hidden layer
    x = Dense(64, kernel_regularizer=Config.REGULARIZATION)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = Dropout(0.2)(x)

    # Output layer for regression on multiple targets
    outputs = Dense(len(Config.TARGET_NAMES), activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def plot_analysis(y_true, y_pred, target_names):
    """Generate plots comparing predictions versus actual values and the error distribution."""
    plt.figure(figsize=(20, 15))

    # Prediction vs Actual for each target
    for i, name in enumerate(target_names):
        plt.subplot(2, 2, i + 1)
        sns.regplot(x=y_true[:, i], y=y_pred[:, i],
                    scatter_kws={'alpha': 0.4}, line_kws={'color': 'red'})
        plt.title(f'{name} - Prediction vs Actual')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        plt.annotate(f'R²: {r2:.3f}', xy=(0.1, 0.85), xycoords='axes fraction')

    # Error distribution
    plt.subplot(2, 2, 3)
    errors = y_true - y_pred
    sns.histplot(errors.flatten(), kde=True)
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error')

    # Correlation matrix of true vs predicted values
    plt.subplot(2, 2, 4)
    combined = pd.DataFrame(np.hstack([y_true, y_pred]),
                            columns=[f'True_{n}' for n in target_names] +
                                    [f'Pred_{n}' for n in target_names])
    sns.heatmap(combined.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')

    plt.tight_layout()
    plt.savefig('light_analysis.png')
    plt.close()


# ===================
# Main Training Logic
# ===================
def main():
    # Load data (adjust paths as needed)
    train_df = load_data('/home/kayode-olalere/PycharmProjects/Project ANN/Training_set/New_Training_set.csv')
    generated_df = load_data(
        '/home/kayode-olalere/PycharmProjects/Project ANN/Codebase/Data_Gen/generated_input_dataset.csv')

    if Config.EXPLORATION_PLOTS:
        # Pairplot of data distribution
        plt.figure(figsize=(20, 20))
        pairplot = sns.pairplot(train_df, diag_kind='kde',
                                plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
                                height=2.5)
        pairplot.fig.suptitle('Data Distribution Pairplot', y=1.02, fontsize=24)
        plt.tight_layout()
        plt.savefig('data_distribution_light.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Correlation heatmap of features
        plt.figure(figsize=(16, 12))
        sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                    annot_kws={'size': 12, 'weight': 'bold'}, fmt='.2f', linewidths=0.5,
                    cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Matrix', fontsize=20, pad=20)
        plt.tight_layout()
        plt.savefig('feature_correlation_light.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Select feature and target columns
    feature_columns = ["l_s", "l_2", "l_1", "s_2", "s_1", "w_s", "w_2", "w_1", "freq"]
    target_columns = Config.TARGET_NAMES

    X = train_df[feature_columns].values.astype('float32')
    y = train_df[target_columns].values.astype('float32')

    # Hold-out split: first split off the test set...
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.SEED
    )
    # ...then carve out a validation set from the training data.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=Config.VAL_SIZE, random_state=Config.SEED
    )

    # Scaling features with RobustScaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'light_scaler.save')

    # Start MLflow run (if you're using MLflow tracking)
    mlflow.set_tracking_uri("mlruns")
    mlflow.tensorflow.autolog()

    with mlflow.start_run():
        # Build and compile the lightweight model
        model = create_light_model(X_train_scaled.shape[1])
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='huber',
                      metrics=[tf.keras.metrics.RootMeanSquaredError()])

        # Define callbacks to prevent over-training and manage learning rate
        callbacks = [
            EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE, restore_best_weights=True),
            ReduceLROnPlateau(patience=Config.REDUCE_LR_PATIENCE, factor=0.5, min_lr=1e-6),
            ModelCheckpoint(Config.CHECKPOINT_PATH, save_best_only=True, monitor='val_loss', mode='min'),
            TensorBoard(log_dir='./logs_light')
        ]

        # Train the model using a simple tf.data pipeline
        train_ds = tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train))
        train_ds = train_ds.batch(Config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val_scaled, y_val))
        val_ds = val_ds.batch(Config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        history = model.fit(train_ds,
                            validation_data=val_ds,
                            epochs=Config.EPOCHS,
                            callbacks=callbacks,
                            verbose=2)

        # Load the best weights from training
        model.load_weights(Config.CHECKPOINT_PATH)

        # Evaluate on the test set
        test_pred = model.predict(X_test_scaled)
        test_mse = mean_squared_error(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)
        print(f"\nFinal Test Metrics:")
        print(f"MSE: {test_mse:.4f}")
        print(f"MAE: {test_mae:.4f}")
        print(f"R²: {test_r2:.4f}  (Accuracy Percentage: {test_r2 * 100:.2f}%)")

        # Generate analysis plots
        plot_analysis(y_test, test_pred, Config.TARGET_NAMES)

        # Optional: SHAP Analysis (can be time-consuming on low-power PCs)
        if Config.RUN_SHAP:
            import shap
            # Use a small sample for SHAP to save time
            explainer = shap.DeepExplainer(model, X_train_scaled[:100])
            shap_values = explainer.shap_values(X_test_scaled[:50])
            plt.figure()
            shap.summary_plot(shap_values, X_test_scaled[:50], feature_names=feature_columns, show=False)
            plt.savefig('light_feature_importance.png')
            plt.close()

        # Generate predictions for new (generated) data
        X_gen = generated_df[feature_columns].values.astype('float32')
        X_gen_scaled = scaler.transform(X_gen)
        gen_pred = model.predict(X_gen_scaled)
        generated_df[target_columns] = gen_pred
        generated_df.to_csv('light_generated_predictions.csv', index=False)

        # Log artifacts if needed
        mlflow.log_artifacts(".")


if __name__ == "__main__":
    main()
