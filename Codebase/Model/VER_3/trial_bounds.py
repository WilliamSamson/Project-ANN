#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

# -------------------------
# Helper functions to load and process data
# -------------------------
def parse_frequency(freq_str):
    """Convert a frequency string (MHz/GHz) into a numeric value in MHz."""
    if not isinstance(freq_str, str):
        try:
            return float(freq_str)
        except Exception:
            return np.nan
    freq_str = freq_str.strip().lower()
    match = re.match(r"([\d.]+)\s*(ghz|mhz)?", freq_str)
    if not match:
        return np.nan
    numeric_part, unit = match.groups()
    try:
        number = float(numeric_part)
    except ValueError:
        return np.nan
    if unit == "ghz":
        return number * 1000  # Convert GHz to MHz.
    else:
        return number

def load_data(path, is_generated=False):
    """
    Load CSV data.

    For training data (is_generated=False), we skip the first row and assign names to columns.
    Expected columns: design parameters [l_s, l_2, l_1, s_2, s_1, w_s, w_2, w_1, freq]
                      followed by targets: [dB(S(1,1)), dB(S(2,1))]
    """
    if is_generated:
        df = pd.read_csv(path, delimiter=",")
        if "ID" in df.columns:
            df = df.drop(columns=["ID"])
    else:
        df = pd.read_csv(
            path,
            skiprows=1,
            header=None,
            names=["l_s", "l_2", "l_1", "s_2", "s_1", "w_s", "w_2", "w_1", "freq", "dB(S(1,1))", "dB(S(2,1))"]
        )
    # Process the frequency column.
    df["freq"] = df["freq"].apply(parse_frequency)
    df["freq"] = pd.to_numeric(df["freq"], errors='coerce')
    return df

# -------------------------
# Analysis function
# -------------------------
def analyze_bounds_for_s11(target=-10.0, tolerance=1.0):
    """
    Loads training data, filters rows where dB(S(1,1)) is within (target ± tolerance) dB,
    and computes the min and max values for each design parameter.

    Also prints the total number of samples found.
    """
    # Find the project root dynamically (go up 4 levels from script directory)
    project_root = Path(__file__).resolve().parents[3]
    data_path = project_root / "Training_set" / "New_Training_set.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")

    # Load the training data.
    train_df = load_data(str(data_path), is_generated=False)

    # Define the tolerance interval.
    lower_bound = target - tolerance
    upper_bound = target + tolerance
    close_df = train_df[(train_df["dB(S(1,1))"] >= lower_bound) & (train_df["dB(S(1,1))"] <= upper_bound)]
    total_found = len(close_df)
    print(f"Total samples with dB(S(1,1)) between {lower_bound} and {upper_bound} dB: {total_found}\n")

    # Define the design parameter columns.
    design_params = ["l_s", "l_2", "l_1", "s_2", "s_1", "w_s", "w_2", "w_1"]

    # Compute and print bounds for each parameter.
    print(f"Design Parameter Bounds (for samples with S11 near {target} dB):")
    for param in design_params:
        param_min = close_df[param].min()
        param_max = close_df[param].max()
        print(f"  {param}: min = {param_min}, max = {param_max}")

    return close_df

# -------------------------
# Main execution
# -------------------------
def main():
    # Prompt user for desired target S11 value and tolerance.
    # Note: The value must be in dB (e.g., -10).
    target_input = input("Enter desired target for dB(S(1,1)) (in dB, note: value must be in dB, e.g., -10): ").strip()
    # Remove any non-numeric characters (except '-' and '.') so that "dB" is stripped out.
    target_input_clean = re.sub(r"[^\d\-.]", "", target_input)
    try:
        target_S11 = float(target_input_clean)
    except ValueError:
        print("Invalid target value entered. Using default of -10.0 dB.")
        target_S11 = -10.0

    tol_input = input("Enter tolerance (in dB, default is 1.0): ").strip()
    try:
        tol = float(tol_input) if tol_input else 1.0
    except ValueError:
        print("Invalid tolerance value entered. Using default of 1.0 dB.")
        tol = 1.0

    analyze_bounds_for_s11(target=target_S11, tolerance=tol)

if __name__ == "__main__":
    main()
