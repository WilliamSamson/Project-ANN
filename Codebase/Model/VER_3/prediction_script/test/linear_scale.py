import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import re
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize, differential_evolution
import concurrent.futures

# Global variable to determine inclusion of dB S21 values
INCLUDE_S21 = True

IMAGE_SAVE_DIR = "Graphs"
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directory for saving graphs
IMAGE_SAVE_DIR = "Graphs"
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# Load Model & Scaler
project_root = Path(__file__).resolve().parents[1]
model_path = project_root / "best_model.pth"
scaler_path = project_root / "scaler.pkl"

if not model_path.exists():
    raise FileNotFoundError(f"File not found: {model_path}")

if not scaler_path.exists():
    raise FileNotFoundError(f"File not found: {scaler_path}")


# Load trained PyTorch model
class ANNModel(nn.Module):
    def __init__(self):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2 if INCLUDE_S21 else 1)  # Output S11 (+ S21 if enabled)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)


# Load trained model
model = ANNModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Set to evaluation mode

# Load Scaler
scaler = joblib.load(scaler_path)


def parse_forward_input(input_str):
    """Parse and validate 9 comma-separated input values."""
    parts = input_str.split(",")
    if len(parts) != 9:
        raise ValueError("Exactly 9 values are required.")
    try:
        values = [float(val) for val in parts[:8]]
        values.append(parse_frequency(parts[8].strip()))
    except ValueError as e:
        raise ValueError(f"Error parsing input: {e}")
    return values


def calculate_accuracy(actual, predicted):
    """Compute accuracy using a normalized error approach."""
    actual = np.array(actual, dtype=np.float32)
    predicted = np.array(predicted, dtype=np.float32)
    denominator = np.maximum(np.maximum(np.abs(actual), np.abs(predicted)), 1e-6)
    error = np.abs((actual - predicted) / denominator)
    mean_error = np.mean(error)
    return 100 * (1 - mean_error)


def parse_frequency(freq_str):
    """Convert frequency string (MHz/GHz) to numeric MHz value."""
    if isinstance(freq_str, str):
        freq_str = freq_str.strip().lower()
        match = re.match(r"([\d.]+)\s*(ghz|mhz)?", freq_str)
        if not match:
            raise ValueError(f"Invalid frequency format: {freq_str}")
        numeric_part, unit = match.groups()
        return float(numeric_part) * 1000 if unit == "ghz" else float(numeric_part)
    else:
        return float(freq_str)


def forward_predict(input_params):
    """Scale input and predict output using the PyTorch model."""
    if len(input_params) != 9:
        raise ValueError(f"Expected 9 features, got {len(input_params)}")

    input_arr = np.array(input_params, dtype=np.float32).reshape(1, -1)
    input_arr_scaled = scaler.transform(input_arr)

    input_tensor = torch.tensor(input_arr_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(input_tensor).cpu().numpy().flatten()

    return pred if INCLUDE_S21 else pred[:1]


def inverse_predict(target, initial_guess):
    """Optimize input parameters to match desired target output."""

    def objective(x):
        pred = forward_predict(x)
        return np.sum((pred - target) ** 2)

    bounds = [
        (7, 13), (6, 25), (6, 25),
        (0.20, 0.6), (0.20, 0.6),
        (0.6, 1.8), (0.5, 2), (0.6, 2),
        (800, 4000)
    ]
    initial_guess = np.clip(initial_guess, [b[0] for b in bounds], [b[1] for b in bounds])
    res = minimize(objective, x0=initial_guess, bounds=bounds, method='L-BFGS-B')

    if not res.success:
        print("\n⚠️ Warning: Optimization may not have converged.")

    optimized_result = res.x
    predicted_output = forward_predict(optimized_result)
    accuracy_percentage = calculate_accuracy(target, predicted_output)

    return optimized_result, res.fun, accuracy_percentage


def global_dual_frequency_optimization(freq1, freq2, bounds):
    """Global optimization using differential evolution."""
    result = differential_evolution(dual_objective, bounds=bounds, args=(freq1, freq2),
                                    strategy='best1bin', maxiter=100, popsize=15,
                                    tol=0.01, mutation=(0.5, 1), recombination=0.7)
    return result.x, result.fun


def format_design_parameters(params):
    """Return a formatted string of design parameters."""
    names = ['l_s', 'l_2', 'l_1', 's_2', 's_1', 'w_s', 'w_2', 'w_1']
    return ", ".join([f"{name}: {float(val):.3f}" for name, val in zip(names, params)])


def plot_performance(initial_params, optimized_params, freq1, freq2):
    """Generate plots for frequency response and S-parameter comparisons."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # Frequency Response Sweep
    freqs = np.linspace(800, 4000, 50)
    predictions = [forward_predict(np.append(np.array(optimized_params), f)) for f in freqs]

    s11 = [p[0] for p in predictions]
    axs[0, 1].plot(freqs, s11, label='dB(S(1,1))')

    if INCLUDE_S21:
        s21 = [p[1] for p in predictions]
        axs[0, 1].plot(freqs, s21, label='dB(S(2,1))')

    axs[0, 1].axvline(freq1, color='r', linestyle='--', label=f'Freq1: {freq1} MHz')
    axs[0, 1].axvline(freq2, color='g', linestyle='--', label=f'Freq2: {freq2} MHz')
    axs[0, 1].set_title('Frequency Response')
    axs[0, 1].set_xlabel('Frequency (MHz)')
    axs[0, 1].set_ylabel('dB')
    axs[0, 1].legend()


    plt.savefig(os.path.join(IMAGE_SAVE_DIR, 'frequency_response.png'))
