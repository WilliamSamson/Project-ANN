import os
import warnings
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import mlflow
from dataclasses import dataclass
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.regularizers import l1_l2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", message=".*DejaVu Sans.*")


@dataclass
class Config:
    SEED: int = 42
    TEST_SIZE: float = 0.20
    VAL_SIZE: float = 0.10
    EPOCHS: int = 100
    BATCH_SIZE: int = 32
    EARLY_STOPPING_PATIENCE: int = 10
    REDUCE_LR_PATIENCE: int = 5
    REGULARIZATION: any = l1_l2(l1=1e-5, l2=1e-4)
    TARGET_NAMES: list = None
    FEATURE_COLUMNS: list = None
    EXPLORATION_PLOTS: bool = True
    CHECKPOINT_PATH: str = "best_light_model.h5"
    RUN_SHAP: bool = False
    TRAIN_DATA_PATH: str = '/home/kayode-olalere/PycharmProjects/Project ANN/Training_set/New_Training_set.csv'
    GENERATED_DATA_PATH: str = '/home/kayode-olalere/PycharmProjects/Project ANN/Codebase/Data_Gen/generated_input_dataset.csv'
    OUTPUT_DIR: str = 'outputs'


# Define expected targets and features.
default_targets = ["dB(S(1,1))", "dB(S(2,1))"]
default_features = ["l_s", "l_2", "l_1", "s_2", "s_1", "w_s", "w_2", "w_1", "freq"]
Config.TARGET_NAMES = default_targets
Config.FEATURE_COLUMNS = default_features

np.random.seed(Config.SEED)
tf.random.set_seed(Config.SEED)


def parse_frequency(freq_str):
    if isinstance(freq_str, str):
        freq_str = freq_str.strip()
        if "MHz" in freq_str:
            return float(freq_str.replace("MHz", "").strip())
        elif "GHz" in freq_str:
            return float(freq_str.replace("GHz", "").strip()) * 1000
    try:
        return float(freq_str)
    except Exception:
        return np.nan


def load_data(path):
    df = pd.read_csv(path)
    # Extract only the first line from each header cell.
    df.columns = [col.splitlines()[0].strip() for col in df.columns]

    if "freq" in df.columns:
        df["freq"] = df["freq"].apply(parse_frequency)
    df.dropna(inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    z = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z < 3).all(axis=1)]
    return df


def create_light_model(input_shape):
    inputs = Input(shape=(input_shape,))
    x = Dense(128, kernel_regularizer=Config.REGULARIZATION)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, kernel_regularizer=Config.REGULARIZATION)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(len(Config.TARGET_NAMES), activation='linear')(x)
    return Model(inputs=inputs, outputs=outputs)


# [Include your plotting functions here as in the previous code.]

def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    train_df = load_data(Config.TRAIN_DATA_PATH)
    generated_df = load_data(Config.GENERATED_DATA_PATH)

    # Validate that all required feature columns exist.
    missing_cols = set(Config.FEATURE_COLUMNS) - set(train_df.columns)
    if missing_cols:
        raise KeyError(
            f"Missing required feature columns: {missing_cols}. Available columns: {train_df.columns.tolist()}")

    # [Rest of your pipeline code: splitting data, scaling, model training, plotting, etc.]


if __name__ == "__main__":
    main()
