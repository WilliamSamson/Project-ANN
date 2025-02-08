import numpy as np
import pandas as pd
from datetime import datetime
import logging
import argparse
from scipy.stats import skew, kurtosis
import statistics
from joblib import Parallel, delayed

# Configure logging to write to both a file and the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data_generation.log"),
        logging.StreamHandler(),
    ],
)

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Generate random dataset with specified parameters.")
    parser.add_argument("--samples", type=int, default=15000, help="Number of samples to generate (default: 15000).")
    parser.add_argument("--precision", type=int, default=2, help="Decimal precision for values (default: 2).")
    parser.add_argument(
        "--distribution",
        type=str,
        choices=["uniform", "gaussian"],
        default="uniform",
        help="Type of distribution to use (default: uniform).",
    )
    return parser.parse_args()

# Function to format frequency values
def format_frequency(val, precision):
    """
    Convert a frequency value (in GHz) to a formatted string.
    If less than 1.0 GHz, it is converted to MHz.
    """
    if val < 1.0:
        return f"{val * 1000:.{precision}f} MHz"
    else:
        return f"{val:.{precision}f} GHz"

# Function to generate random data using parallelism
def generate_data_parallel(parameter_ranges, num_samples, use_gaussian, precision):
    """
    Generate random data in parallel for given parameter ranges.

    Args:
        parameter_ranges (dict): Ranges for parameters (low, high).
        num_samples (int): Number of samples to generate.
        use_gaussian (bool): Whether to use Gaussian distribution.
        precision (int): Decimal precision for generated values.

    Returns:
        dict: Dictionary of generated parameter data.
    """
    def generate_param(param, low, high):
        if use_gaussian:
            mean = (low + high) / 2
            std_dev = (high - low) / 6  # ~99.7% of data within [low, high]
            values = np.random.normal(mean, std_dev, num_samples)
            clipped_count = np.sum((values < low) | (values > high))
            if clipped_count > 0:
                logging.warning(f"Clipping {clipped_count} values for parameter '{param}' due to range mismatch.")
            values = np.clip(values, low, high)
        else:
            values = np.random.uniform(low, high, num_samples)
        # Round numeric values first
        values = np.round(values, precision)
        # If the parameter is "freq", format the values as strings with units.
        if param == "freq":
            values = np.vectorize(lambda x: format_frequency(x, precision))(values)
        return values

    results = Parallel(n_jobs=-1)(
        delayed(generate_param)(param, low, high) for param, (low, high) in parameter_ranges.items()
    )
    return {param: result for param, result in zip(parameter_ranges.keys(), results)}

# Function to save a DataFrame to a CSV file
def save_csv(dataframe, filename_prefix):
    """
    Save a DataFrame to a CSV file with a timestamped filename.

    Args:
        dataframe (pd.DataFrame): DataFrame to save.
        filename_prefix (str): Prefix for the file name.

    Returns:
        str: The full filename of the saved CSV.
    """
    filename = f"{filename_prefix}.csv"
    try:
        dataframe.to_csv(filename, index=False)
        logging.info(f"File saved successfully: {filename}")
    except IOError as e:
        logging.error(f"Failed to save file '{filename}': {e}")
        raise
    return filename

# Function to compute additional metrics
def compute_additional_metrics(dataframe):
    """
    Compute additional metrics like skewness, kurtosis, and mode.

    Args:
        dataframe (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with additional metrics.
    """
    metrics = {}
    # Skip the "ID" column (assumed to be the first column)
    for column in dataframe.columns[1:]:
        metrics[column] = {
            "Skewness": skew(dataframe[column]) if dataframe[column].dtype != object else None,
            "Kurtosis": kurtosis(dataframe[column]) if dataframe[column].dtype != object else None,
            "Mode": statistics.mode(dataframe[column]) if dataframe[column].dtype != object else None,
        }
    return pd.DataFrame(metrics).T

# Main script execution
if __name__ == "__main__":
    args = parse_args()

    # Updated parameter ranges and headers (excluding s1 and s2)
    PARAMETER_RANGES = {
        "l_s": (5.5, 6.5),    # e.g., length start (mm)
        "l_2": (6.5, 7.5),    # e.g., secondary length (mm)
        "l_1": (6.5, 7.5),    # e.g., primary length (mm)
        "w_s": (0.5, 0.7),    # e.g., width start (mm)
        "w_2": (0.4, 0.6),    # e.g., secondary width (mm)
        "w_1": (0.1, 0.3),    # e.g., primary width (mm)
        "freq": (0.8, 4.0)    # Frequency in GHz
    }

    # Validate input parameters
    if args.samples <= 0:
        raise ValueError("The number of samples must be greater than 0.")
    for param, (low, high) in PARAMETER_RANGES.items():
        if low >= high:
            raise ValueError(f"Invalid range for parameter '{param}': low >= high.")

    # Generate random data
    try:
        random_data = generate_data_parallel(
            PARAMETER_RANGES, args.samples, args.distribution == "gaussian", args.precision
        )

        # Create DataFrame and add unique ID column
        dataset = pd.DataFrame(random_data)
        dataset.insert(0, "ID", range(1, args.samples + 1))

        # Save dataset to CSV
        output_filename = save_csv(dataset, "generated_input_dataset")
        logging.info(f"Dataset with {args.samples} samples generated and saved to '{output_filename}'.")

        # Save summary statistics
        summary_stats = dataset.describe().T
        summary_stats_filename = save_csv(summary_stats, "summary_statistics")
        logging.info(f"Summary statistics saved to '{summary_stats_filename}'.")

        # Save additional metrics
        additional_metrics = compute_additional_metrics(dataset)
        additional_metrics_filename = save_csv(additional_metrics, "additional_metrics")
        logging.info(f"Additional metrics saved to '{additional_metrics_filename}'.")

        logging.info("Data generation complete with precision and accuracy maintained.")
    except Exception as e:
        logging.error(f"Error occurred during data generation: {e}")
