import os
# Suppress TensorFlow informational messages (including AVX2/FMA info)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
# Suppress warnings related to missing DejaVu Sans fonts
warnings.filterwarnings("ignore", message=".*DejaVu Sans.*")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import shap
import mlflow
import joblib

# Enable mixed precision training (requires TF 2.4+ and compatible hardware)
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')


# Configuration Class
class Config:
    SEED = 42
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    N_SPLITS = 5
    EPOCHS = 500
    BATCH_SIZE = 32
    EARLY_STOPPING_PATIENCE = 25
    REDUCE_LR_PATIENCE = 12
    REGULARIZATION = l1_l2(l1=1e-5, l2=1e-4)
    # Update target names to match your new training set
    TARGET_NAMES = ["dB(S(1,1))", "dB(S(2,1))"]
    EXPLORATION_PLOTS = True


# Reproducibility
np.random.seed(Config.SEED)
tf.random.set_seed(Config.SEED)


def parse_frequency(freq_str):
    """
    Convert frequency string to a float in MHz.
    For example, '800.0 MHz' becomes 800.0 and '1.000 GHz' becomes 1000.0.
    """
    if isinstance(freq_str, str):
        freq_str = freq_str.strip()
        if "MHz" in freq_str:
            return float(freq_str.replace("MHz", "").strip())
        elif "GHz" in freq_str:
            # Convert GHz to MHz (1 GHz = 1000 MHz)
            return float(freq_str.replace("GHz", "").strip()) * 1000
    return float(freq_str)


def load_data(path):
    """Load and preprocess data with outlier detection."""
    df = pd.read_csv(path)

    # Convert frequency column if it exists
    if "freq" in df.columns:
        df["freq"] = df["freq"].apply(parse_frequency)

    # Remove outliers using Z-score on all numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    z = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z < 3).all(axis=1)]

    return df


def create_model(input_shape):
    """Create advanced dual-path neural network with residual connections."""
    inputs = Input(shape=(input_shape,))

    # Feature extraction path
    x = Dense(512, kernel_regularizer=Config.REGULARIZATION)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)

    # Residual block
    residual = Dense(256)(x)
    x = Dense(256, kernel_regularizer=Config.REGULARIZATION)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)
    x = concatenate([x, residual])

    # Output path
    x = Dense(128, kernel_regularizer=Config.REGULARIZATION)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.2)(x)

    outputs = Dense(len(Config.TARGET_NAMES))(x)
    # Cast output to float32 to maintain loss stability with mixed precision
    outputs = tf.keras.layers.Activation('linear', dtype='float32')(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def plot_analysis(y_true, y_pred, target_names):
    """Create comprehensive analysis plots."""
    plt.figure(figsize=(20, 15))

    # Prediction vs Actual
    for i, name in enumerate(target_names):
        plt.subplot(2, 2, i + 1)
        sns.regplot(x=y_true[:, i], y=y_pred[:, i],
                    scatter_kws={'alpha': 0.4}, line_kws={'color': 'red'})
        plt.title(f'{name} - Prediction vs Actual')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        plt.annotate(f'R²: {r2:.3f}', xy=(0.1, 0.85), xycoords='axes face')

    # Error Distribution
    plt.subplot(2, 2, 3)
    errors = y_true - y_pred
    sns.histplot(errors.flatten(), kde=True)
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error')

    # Correlation Matrix
    plt.subplot(2, 2, 4)
    combined = pd.DataFrame(np.hstack([y_true, y_pred]),
                            columns=[f'True_{n}' for n in target_names] +
                                    [f'Pred_{n}' for n in target_names])
    sns.heatmap(combined.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')

    plt.tight_layout()
    plt.savefig('advanced_analysis.png')
    plt.close()


def main():
    # Update training set file path to use your new training set
    train_df = load_data('/home/kayode-olalere/PycharmProjects/Project ANN/Training_set/New_Training_set.csv')
    generated_df = load_data('/home/kayode-olalere/PycharmProjects/Project ANN/Codebase/Data_Gen/generated_input_dataset.csv')

    if Config.EXPLORATION_PLOTS:
        # Pairplot for data distribution
        plt.figure(figsize=(20, 20))
        pairplot = sns.pairplot(train_df, diag_kind='kde',
                                plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
                                height=2.5)
        pairplot.fig.suptitle('Data Distribution Pairplot', y=1.02, fontsize=24)
        for ax in pairplot.axes.flatten():
            ax.set_xlabel(ax.get_xlabel(), fontsize=14)
            ax.set_ylabel(ax.get_ylabel(), fontsize=14)
            ax.tick_params(labelsize=12)
        plt.tight_layout()
        plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Correlation heatmap
        plt.figure(figsize=(16, 12))
        sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                    annot_kws={'size': 12, 'weight': 'bold'}, fmt='.2f', linewidths=0.5,
                    cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Matrix', fontsize=20, pad=20)
        plt.xticks(fontsize=14, rotation=45, ha='right')
        plt.yticks(fontsize=14, rotation=0)
        plt.tight_layout()
        plt.savefig('feature_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Data Preparation: select features and target columns
    feature_columns = ["l_s", "l_2", "l_1", "s_2", "s_1", "w_s", "w_2", "w_1", "freq"]
    target_columns = Config.TARGET_NAMES  # ["dB(S(1,1))", "dB(S(2,1))"]

    X = train_df[feature_columns].values.astype('float32')
    y = train_df[target_columns].values.astype('float32')

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.SEED
    )

    # Scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'advanced_scaler.save')

    # MLflow Tracking
    mlflow.set_tracking_uri("mlruns")
    mlflow.tensorflow.autolog()

    with mlflow.start_run():
        # Build the model
        model = create_model(X_train_scaled.shape[1])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

        # Callbacks
        callbacks = [
            EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE,
                          restore_best_weights=True),
            ReduceLROnPlateau(patience=Config.REDUCE_LR_PATIENCE,
                              factor=0.5, min_lr=1e-6),
            ModelCheckpoint(
                '/home/kayode-olalere/PycharmProjects/Project ANN/Codebase/Training_set/VER_3/v_1/model_advanced.h5',
                save_best_only=True),
            TensorBoard(log_dir='./logs')
        ]

        # Training with K-Fold Cross Validation using tf.data for speed
        kf = KFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.SEED)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
            print(f"\nFold {fold + 1}/{Config.N_SPLITS}")
            X_train_fold, X_val_fold = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            # Create tf.data.Dataset pipelines for training and validation
            train_ds = tf.data.Dataset.from_tensor_slices((X_train_fold, y_train_fold))
            train_ds = train_ds.shuffle(buffer_size=1024).batch(Config.BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

            val_ds = tf.data.Dataset.from_tensor_slices((X_val_fold, y_val_fold))
            val_ds = val_ds.batch(Config.BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=Config.EPOCHS,
                callbacks=callbacks,
                verbose=2
            )

            val_score = model.evaluate(val_ds, verbose=0)
            fold_scores.append(val_score[0])
            mlflow.log_metric(f"fold_{fold + 1}_loss", val_score[0])

        # Final Evaluation using numpy arrays for prediction
        model.load_weights('best_model_advanced.h5')
        test_pred = model.predict(X_test_scaled)

        test_mse = mean_squared_error(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)
        accuracy_percentage = test_r2 * 100

        print(f"\nFinal Test Metrics:")
        print(f"MSE: {test_mse:.4f}")
        print(f"MAE: {test_mae:.4f}")
        print(f"R² (Accuracy %): {accuracy_percentage:.2f}%")

        # Generate analysis plots
        plot_analysis(y_test, test_pred, Config.TARGET_NAMES)

        # SHAP Analysis (using a sample of the training data)
        explainer = shap.DeepExplainer(model, X_train_scaled[:1000])
        shap_values = explainer.shap_values(X_test_scaled[:100])
        plt.figure()
        shap.summary_plot(shap_values, X_test_scaled[:100], feature_names=feature_columns)
        plt.savefig('feature_importance.png')
        plt.close()

        # Generate Predictions for New Data using the same feature columns
        X_gen = generated_df[feature_columns].values.astype('float32')
        X_gen_scaled = scaler.transform(X_gen)
        gen_pred = model.predict(X_gen_scaled)
        generated_df[target_columns] = gen_pred
        generated_df.to_csv('advanced_generated_predictions.csv', index=False)

        # Log artifacts with MLflow
        mlflow.log_artifacts("..")


if __name__ == "__main__":
    main()
