import pandas as pd
import os

# Define the path where you want to save the formatted data
formatted_data_path = '/home/kayode-olalere/PycharmProjects/Project ANN/Model/Formatted_Training_Data.csv'

# Create the directory if it does not exist
os.makedirs(os.path.dirname(formatted_data_path), exist_ok=True)

# Load the original training data
training_data_path = '/home/kayode-olalere/PycharmProjects/Project ANN/Trainning Dataset.csv'

# Load the CSV without specifying column names initially to inspect
df = pd.read_csv(training_data_path)

# Check the first few rows to understand the structure
print(df.head())

# Manually assign column names based on your input data format
df.columns = ['S/N', 'W1 (mm)', 'L1 (mm)', 'D1 (mm)', 'W2 (mm)', 'L2 (mm)', 'Fc (GHz)', 'BW (GHz)', 'SF (dB)', 'Ro (dB/GHz)']

# Now remove the 'BW (GHz)' column
df_cleaned = df.drop(columns=['BW (GHz)'])

# Rename the columns as per the desired format
df_cleaned.rename(columns={
    'S/N': 'ID',
    'Fc (GHz)': 'Frequency (GHz)',
    'W1 (mm)': 'W1 (mm)',
    'L1 (mm)': 'L1 (mm)',
    'D1 (mm)': 'D1 (mm)',
    'W2 (mm)': 'W2 (mm)',
    'L2 (mm)': 'L2 (mm)',
    'SF (dB)': 'S1',           # Rename SF to S1
    'Ro (dB/GHz)': 'S2'        # Rename Ro to S2
}, inplace=True)

# Reorder the columns as per the required output structure
df_cleaned = df_cleaned[['ID', 'Frequency (GHz)', 'W1 (mm)', 'L1 (mm)', 'D1 (mm)', 'W2 (mm)', 'L2 (mm)', 'S1', 'S2']]

# Save the cleaned and formatted data
df_cleaned.to_csv(formatted_data_path, index=False)

# Show a preview of the cleaned data
print(df_cleaned.head())
