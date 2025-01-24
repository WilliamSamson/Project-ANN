import numpy as np
import pandas as pd
from datetime import datetime

# Seed for reproducibility (optional)
SEED = 42
np.random.seed(SEED)

# Parameters for dataset generation
NUM_SAMPLES = 15000  # Number of rows (samples) to generate

# Define realistic ranges for each parameter
PARAMETER_RANGES = {
    "Frequency (GHz)": (0.5, 5.0),  # GHz
    "W1 (mm)": (0.1, 5.0),  # mm (width 1)
    "L1 (mm)": (0.5, 20.0),  # mm (length 1)
    "D1 (mm)": (0.1, 2.0),  # mm (distance or spacing)
    "W2 (mm)": (0.1, 5.0),  # mm (width 2)
    "L2 (mm)": (0.5, 20.0),  # mm (length 2)
}

# Option to use Gaussian distribution instead of uniform
USE_GAUSSIAN = False  # Set to True for Gaussian distribution


def generate_random_data(num_samples, parameter_ranges, use_gaussian, precision=4):
    """
    Generate random data for given parameter ranges.

    Args:
        num_samples (int): Number of samples to generate.
        parameter_ranges (dict): Ranges for parameters (low, high).
        use_gaussian (bool): Whether to use Gaussian distribution.
        precision (int): Decimal precision for generated values.

    Returns:
        dict: Dictionary of generated parameter data.
    """
    data = {}
    for param, (low, high) in parameter_ranges.items():
        if use_gaussian:
            mean = (low + high) / 2
            std_dev = (high - low) / 6  # ~99.7% of data within range
            values = np.clip(np.random.normal(mean, std_dev, num_samples), low, high)
        else:
            values = np.random.uniform(low, high, num_samples)

        # Cap values to the specified precision
        data[param] = np.round(values, precision)

    return data


def save_csv(dataframe, filename_prefix):
    """
    Save a DataFrame to a CSV file with a timestamped filename.

    Args:
        dataframe (pd.DataFrame): DataFrame to save.
        filename_prefix (str): Prefix for the file name.

    Returns:
        str: The full filename of the saved CSV.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{filename_prefix}_{timestamp}.csv"
    dataframe.to_csv(filename, index=False)
    return filename


# Generate random values for each parameter
random_data = generate_random_data(NUM_SAMPLES, PARAMETER_RANGES, USE_GAUSSIAN)

# Create a DataFrame and add a unique ID column
dataset = pd.DataFrame(random_data)
dataset.insert(0, "ID", range(1, NUM_SAMPLES + 1))

# Save the dataset to a CSV file
output_filename = save_csv(dataset, "generated_input_dataset")
print(f"Dataset with {NUM_SAMPLES} samples generated and saved to '{output_filename}'.")

# Generate and save summary statistics
summary_stats = dataset.describe().T
summary_stats_output = save_csv(summary_stats, "summary_statistics")
print(f"Summary statistics saved to '{summary_stats_output}'.")

# Final confirmation
print("Data generation complete with precision and accuracy maintained.")
