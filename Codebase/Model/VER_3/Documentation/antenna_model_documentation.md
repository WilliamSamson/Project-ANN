**Comprehensive Documentation for ANN Training Pipeline**

## **1. Introduction**  
This document explains an end-to-end pipeline for training an Artificial Neural Network (ANN) to predict RF circuit parameters. The goal is to help even a beginner understand how the code works, why specific implementations were chosen, and how they contribute to the overall model performance.  

---

## **2. Overview of the Code**  

The script performs the following steps:  

1. **Configuration Setup** – Defines model hyperparameters and dataset details.  
2. **Data Loading & Processing** – Reads data, cleans it, and prepares it for training.  
3. **Data Splitting & Scaling** – Splits data into training, validation, and test sets while ensuring proper scaling.  
4. **Model Architecture Definition** – Constructs a deep learning model using TensorFlow/Keras.  
5. **Training Process** – Trains the model with monitoring mechanisms like early stopping and learning rate reduction.  
6. **Evaluation & Visualization** – Assesses model performance using metrics and generates plots.  

---

## **3. Key Components and Explanations**  

### **3.1 Configuration Parameters (`Config` Class)**  
This class contains all hyperparameters and configurations used throughout the training pipeline.  


```
python
class Config:
## SEED = 42
## TEST_SIZE = 0.20
## VAL_SIZE = 0.10
## EPOCHS = 50
## BATCH_SIZE = 128
## EARLY_STOPPING_PATIENCE = 10
## REDUCE_LR_PATIENCE = 5
    REGULARIZATION = l1_l2(l1=1e-5, l2=1e-4)  
    FEATURE_COLS = ["l_s", "l_2", "l_1", "s_2", "s_1", "w_s", "w_2", "w_1", "freq"]  
    TARGET_COLS = ["dB(S(1,1))", "dB(S(2,1))"]  
    CHECKPOINT_PATH = "best_model.h5"
```


### **Why These Choices?**  
- `SEED = 42`: Ensures reproducibility across multiple runs.  
- `TEST_SIZE = 0.20`: 20% of data is reserved for final model evaluation.  
- `VAL_SIZE = 0.10`: 10% of training data is used for validation during training.  
- `EPOCHS` = 50  # Number of times the model will iterate over the entire training dataset. Higher values improve learning but may lead to overfitting if too large.
- `BATCH_SIZE = 128`: Processes 128 samples at a time for efficient training.
- `EARLY_STOPPING_PATIENCE = 10`: Stops training if validation loss doesn’t improve for 10 epochs.  
- `REDUCE_LR_PATIENCE = 5`: Reduces learning rate if performance stagnates for 5 epochs.  
- `REGULARIZATION`: Uses L1-L2 regularization to prevent overfitting.  

---

### **3.2 Data Handling**  

#### **Parsing Frequency Data (`parse_frequency`)**  
The function ensures that all frequency values are converted to MHz regardless of input format.  


```
python
def parse_frequency(freq_str):
    if not isinstance(freq_str, str):
        try:
            return float(freq_str)
        except Exception:
            return np.nan

    freq_str = freq_str.replace("\xa0", " ").strip().strip('"').strip("'")
    lower_str = freq_str.lower()
    numeric_part = re.sub(r'[^0-9\.]', '', freq_str)

    if numeric_part == "":
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
        return number * 1000  
    elif "mhz" in lower_str:
        return number  
    else:
        return number
```


**Why This?**  
- Some datasets contain frequency values in different formats (GHz, MHz).  
- Ensures consistency by converting all values to MHz.  
- Handles missing or incorrect data gracefully.  

---

#### **Loading Data (`load_data`)**  
Reads the dataset, applies the frequency parser, and removes outliers.  


```
python
def load_data(path, is_generated=False):
    if is_generated:
        df = pd.read_csv(path, delimiter=",")
        if "ID" in df.columns:
            df = df.drop(columns=["ID"])
    else:
        df = pd.read_csv(
            path, skiprows=1, header=None, names=Config.FEATURE_COLS + Config.TARGET_COLS
        )

    df["freq"] = df["freq"].apply(parse_frequency)
    df["freq"] = pd.to_numeric(df["freq"], errors='coerce')

    if not is_generated:
        numeric_cols = Config.FEATURE_COLS + Config.TARGET_COLS
        z = np.abs(stats.zscore(df[numeric_cols]))
        df = df[(z < 3).all(axis=1)]

    return df
```


**Why This?**  
- Drops irrelevant columns (like "ID").  
- Skips metadata rows in training datasets.  
- Removes extreme outliers using the Z-score method (threshold = 3).  

---

### **3.3 Model Definition (`create_model`)**  
Constructs the ANN architecture.  


```
python
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
```


**Why This?**  
- **256 & 128 neurons:** Empirically chosen for balanced performance.  
- **Batch Normalization:** Stabilizes training and accelerates convergence.  
- **Dropout (0.3):** Reduces overfitting by randomly deactivating 30% of neurons.  

---

### **3.4 Training Process (`main`)**  
Handles dataset preparation, training, and evaluation.  


```
python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE, random_state=Config.SEED)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=Config.VAL_SIZE, random_state=Config.SEED)
```


**Why?**  
- Splits data into train (70%), validation (10%), and test (20%) sets for unbiased evaluation.  


```
python
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl')
```


**Why?**  
- **RobustScaler:** Handles outliers better than MinMax or Standard Scaler.  

---

### **3.5 Evaluation and Performance Reporting**  
After training, the script evaluates performance and generates plots.  


```
python
r2_scores = [r2_score(y_test[:, i], test_pred[:, i]) for i in range(2)]
mae_scores = [mean_absolute_error(y_test[:, i], test_pred[:, i]) for i in range(2)]
```


**Why?**  
- **R² Score:** Measures model accuracy (closer to 1 is better).  
- **MAE:** Shows average prediction error in decibels (lower is better).  

---

## **4. Summary of Key Learnings**  
- Uses **deep learning** for RF circuit parameter prediction.  
- **Data preprocessing** ensures clean input.  
- **Batch normalization and dropout** help with training stability and generalization.  
- **Early stopping & learning rate reduction** prevent wasted computation.  
