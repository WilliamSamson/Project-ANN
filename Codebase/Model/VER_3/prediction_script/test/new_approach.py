import os
import numpy as np
import tensorflow as tf
import joblib
import re
import logging
from pathlib import Path
from scipy.optimize import minimize, differential_evolution
import concurrent.futures

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
INCLUDE_S21 = True
IMAGE_SAVE_DIR = "Graphs"
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# Get Project Root
try:
    project_root = Path(__file__).resolve().parents[2]
except NameError:
    project_root = Path(os.getcwd())  # Fallback for interactive environments

model_path = project_root / "best_model.h5"
scaler_path = project_root / "scaler.pkl"

if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not scaler_path.exists():
    raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

class ModelHandler:
    """Handles model loading, predictions, and scaling operations."""

    def __init__(self, model_path, scaler_path):
        self.model = self._load_model(model_path)
        self.scaler = self._load_scaler(scaler_path)

    def _load_model(self, path):
        return tf.keras.models.load_model(str(path), compile=False)

    def _load_scaler(self, path):
        return joblib.load(str(path))

    def predict(self, input_params):
        if len(input_params) != 9:
            raise ValueError(f"Expected 9 features, got {len(input_params)}")
        input_scaled = self.scaler.transform(np.array(input_params, dtype=np.float32).reshape(1, -1))
        prediction = self.model(input_scaled, training=False).numpy().flatten()
        return prediction if INCLUDE_S21 else prediction[:1]  # Return S11 only if S21 is excluded

def evaluate_bounds(bounds, freq1, freq2, model_handler):
    """Evaluates different bounds configurations."""
    def dual_objective(x):
        target = np.array([-15, -1, -15, -1] if INCLUDE_S21 else [-15, -15], dtype=np.float32)
        pred_f1 = model_handler.predict(np.append(x, freq1))
        pred_f2 = model_handler.predict(np.append(x, freq2))
        return np.sum((np.concatenate([pred_f1, pred_f2]) - target) ** 2)

    try:
        # Global Optimization
        global_result = differential_evolution(dual_objective, bounds=bounds, strategy='best1bin', maxiter=100, popsize=15, tol=0.01)
        global_candidate, global_error = global_result.x, global_result.fun

        # Local Optimization
        initial_guess = np.mean(bounds, axis=1)
        local_result = minimize(dual_objective, x0=initial_guess, bounds=bounds, method='L-BFGS-B')
        local_candidate, local_error = local_result.x, local_result.fun

        return bounds, global_candidate, global_error, local_candidate, local_error
    except Exception as e:
        logging.error(f"Optimization failed: {e}")
        return None

def run_possible_bounds(freq1, freq2, model_handler):
    """Tests multiple bounds configurations in parallel."""
    possible_bounds = [
        [(6, 13), (7, 25), (7, 25), (0.15, 0.6), (0.15, 0.6), (0.6, 1.8), (0.5, 2.0), (0.2, 2.0)],
        [(7, 13), (6, 25), (6, 25), (0.20, 0.6), (0.20, 0.6), (0.6, 1.8), (0.5, 2.0), (0.6, 2.0)]
    ]

    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:  # Fixed pickling issue
        futures = {executor.submit(evaluate_bounds, b, freq1, freq2, model_handler): b for b in possible_bounds}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    best = min(results, key=lambda x: x[2])  # Select best based on global error
    return best[0], best[3]  # Return best bounds & local candidate

if __name__ == "__main__":
    # Load Model
    model_handler = ModelHandler(model_path, scaler_path)

    # Run Optimization
    try:
        best_bounds, best_candidate = run_possible_bounds(900, 2500, model_handler)
        logging.info(f"Best Bounds: {best_bounds}")
        logging.info(f"Optimized Candidate: {best_candidate}")
    except Exception as e:
        logging.error(f"Execution failed: {e}")
