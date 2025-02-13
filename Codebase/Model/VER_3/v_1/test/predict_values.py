import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.optimize import minimize
import joblib
import re

# Load Model & Scaler
model = load_model('/home/kayode-olalere/PycharmProjects/Project ANN/Codebase/Model/VER_3/v_1/best_model.h5', compile=False)
scaler = joblib.load('/home/kayode-olalere/PycharmProjects/Project ANN/Codebase/Model/VER_3/v_1/scaler.pkl')

def parse_frequency(freq_str):
    """ Convert frequency string (MHz/GHz) to numeric MHz value. """
    if isinstance(freq_str, str):
        freq_str = freq_str.strip().lower()
        match = re.match(r"([\d.]+)\s*(ghz|mhz)?", freq_str)
        if not match:
            raise ValueError(f"Invalid frequency format: {freq_str}")

        numeric_part, unit = match.groups()
        return float(numeric_part) * 1000 if unit == "ghz" else float(numeric_part)
    else:
        return float(freq_str)

def parse_forward_input(input_str):
    """ Parse and validate 9 comma-separated input values. """
    parts = input_str.split(",")
    if len(parts) != 9:
        raise ValueError("Exactly 9 values are required.")

    try:
        values = [float(val) for val in parts[:8]]
        values.append(parse_frequency(parts[8].strip()))
    except ValueError as e:
        raise ValueError(f"Error parsing input: {e}")

    return values

def forward_predict(input_params):
    """ Scale input and predict output using the model. """
    input_arr = np.array(input_params, dtype='float32').reshape(1, -1)
    input_arr_scaled = scaler.transform(input_arr)
    pred = model(input_arr_scaled, training=False)
    return pred.numpy().flatten()

def calculate_accuracy(actual, predicted):
    """ Compute accuracy using a normalized error approach (FOR INVERSE MODE ONLY). """
    actual = np.array(actual, dtype=np.float32)
    predicted = np.array(predicted, dtype=np.float32)

    # Fixed denominator calculation with nested maximums
    denominator = np.maximum(np.maximum(np.abs(actual), np.abs(predicted)), 1e-6)
    error = np.abs((actual - predicted) / denominator) * 100
    mean_error = np.mean(error)

    accuracy_percentage = max(0, 100 - min(mean_error, 100))

    if accuracy_percentage >= 96:
        stars = "⭐⭐⭐⭐⭐ (5/5)"
    elif accuracy_percentage >= 86:
        stars = "⭐⭐⭐⭐ (4/5)"
    elif accuracy_percentage >= 71:
        stars = "⭐⭐⭐ (3/5)"
    elif accuracy_percentage >= 51:
        stars = "⭐⭐ (2/5)"
    else:
        stars = "⭐ (1/5)"

    return accuracy_percentage, stars

def inverse_predict(target, initial_guess):
    """ Optimize input parameters to match desired target output. """
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
    accuracy_percentage, star_rating = calculate_accuracy(target, predicted_output)

    return optimized_result, res.fun, accuracy_percentage, star_rating

def main():
    print("Welcome to the Enhanced Prediction System!")
    print("Choose a mode:")
    print("1. Forward Prediction")
    print("2. Inverse Prediction")
    mode = input("Enter mode (forward/inverse): ").strip().lower()

    if mode == "forward":
        print("\nForward Prediction Mode:")
        print("Enter 9 comma-separated values [l_s, l_2, l_1, s_2, s_1, w_s, w_2, w_1, freq]")
        input_str = input("Input: ")

        try:
            input_params = parse_forward_input(input_str)
            predicted = forward_predict(input_params)
            # Removed accuracy calculation for forward mode
            print(f"Model predicted output: {predicted}")

        except ValueError as e:
            print(f"Error: {e}")

    elif mode == "inverse":
        print("\nInverse Prediction Mode:")
        print("Enter desired dB values for [dB(S(1,1)), dB(S(2,1))] (comma-separated)")
        target_str = input("Target: ")

        try:
            target = np.array([float(val) for val in target_str.split(",")])
            if target.shape[0] != 2:
                raise ValueError("Exactly 2 target values are required.")
        except ValueError as e:
            print(f"Error: {e}")
            return

        print("Enter an initial guess for 9 input parameters or press Enter for default.")
        guess_str = input("Initial guess: ").strip()
        bounds = [
            (7, 13), (6, 25), (6, 25),
            (0.20, 0.6), (0.20, 0.6),
            (0.6, 1.8), (0.5, 2), (0.6, 2),
            (800, 4000)
        ]

        if not guess_str:
            initial_guess = [(lb + ub) / 2 for lb, ub in bounds]
            print(f"Using default guess: {initial_guess}")
        else:
            try:
                initial_guess = parse_forward_input(guess_str)
            except ValueError as e:
                print(f"Error: {e}")
                return

        solution, error_val, accuracy_percentage, star_rating = inverse_predict(target, initial_guess)

        formatted_values = [f"{val:.3f}" for val in solution[:-1]]
        freq_display = f"{solution[-1] / 1000:.3f} GHz" if solution[-1] >= 1000 else f"{solution[-1]:.3f} MHz"
        formatted_solution = ", ".join(formatted_values + [freq_display])

        print("\nOptimized Input Parameters:")
        print(formatted_solution)
        print(f"Final objective (error value): {error_val:.6f}")
        print(f"🔹 Accuracy: {accuracy_percentage:.2f}%  {star_rating}")

    else:
        print("Invalid mode. Please restart and choose 'forward' or 'inverse'.")

if __name__ == "__main__":
    main()