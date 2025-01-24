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
    parser.add_argument("--precision", type=int, default=4, help="Decimal precision for values (default: 4).")
    parser.add_argument(
        "--distribution",
        type=str,
        choices=["uniform", "gaussian"],
        default="uniform",
        help="Type of distribution to use (default: uniform).",
    )
    return parser.parse_args()

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
            std_dev = (high - low) / 6
            values = np.random.normal(mean, std_dev, num_samples)
            clipped_count = np.sum((values < low) | (values > high))
            if clipped_count > 0:
                logging.warning(f"Clipping {clipped_count} values for parameter '{param}' due to range mismatch.")
            values = np.clip(values, low, high)
        else:
            values = np.random.uniform(low, high, num_samples)
        return np.round(values, precision)

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
        str: The full filename of the saved CSV, with a timestamp appended.

    Raises:
        IOError: If the file cannot be saved.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.csv"
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
    for column in dataframe.columns[1:]:  # Skip "ID"
        metrics[column] = {
            "Skewness": skew(dataframe[column]),
            "Kurtosis": kurtosis(dataframe[column]),
            "Mode": statistics.mode(dataframe[column]),
        }
    return pd.DataFrame(metrics).T

# Main script execution
if __name__ == "__main__":
    args = parse_args()

    # Parameter ranges
    PARAMETER_RANGES = {
        "Frequency (GHz)": (0.5, 5.0),  # GHz
        "W1 (mm)": (0.1, 5.0),  # mm (width 1)
        "L1 (mm)": (0.5, 20.0),  # mm (length 1)
        "D1 (mm)": (0.1, 2.0),  # mm (distance or spacing)
        "W2 (mm)": (0.1, 5.0),  # mm (width 2)
        "L2 (mm)": (0.5, 20.0),  # mm (length 2)
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
