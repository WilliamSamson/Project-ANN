import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Step 1: Load the trained model
model = load_model("/home/kayode-olalere/PycharmProjects/Project ANN/Codebase/Training_set/VER_3/model_advanced.h5")

# Step 2: Load training data to fit the scaler
training_data_path = '/home/kayode-olalere/PycharmProjects/Project ANN/Training_set/New_Training_set.csv'
df = pd.read_csv(training_data_path)

# Ensure columns are numeric by skipping headers and irrelevant columns (like S/N)
# Assuming the input features are columns 1 through 6 (not just 5)
X_train = df.iloc[:, 1:7].apply(pd.to_numeric, errors='coerce').dropna().values  # 6 features now

# Step 3: Fit the scaler on the training data
scaler = StandardScaler()
scaler.fit(X_train)  # Fit the scaler on the input features

# Step 4: Define new input data (with 6 features)
new_data = np.array([[6.000,7.000,7.000,0.150,0.150,0.600,0.500,0.200,800.0,-0.471,-9.909]])  # 6 features, adjusted

# Scale the new data
new_data_scaled = scaler.transform(new_data)

# Step 5: Make predictions
predictions = model.predict(new_data_scaled)
print("Predictions:", predictions)
