import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.optimize import minimize
import joblib
import re

# Load the pre-trained model (for inference, so compile=False)
model = load_model('/home/kayode-olalere/PycharmProjects/Project ANN/Codebase/Model/VER_3/v_1/best_model.h5',
                   compile=False)

# Load the scaler that was used during training
scaler = joblib.load('/home/kayode-olalere/PycharmProjects/Project ANN/Codebase/Model/VER_3/v_1/scaler.pkl')


def parse_frequency(freq_str):
    """ Convert a frequency string (MHz/GHz) into a numeric MHz value with validation. """
    if isinstance(freq_str, str):
        freq_str = freq_str.strip().lower()
        match = re.match(r"([\d.]+)\s*(ghz|mhz)?", freq_str)
        if not match:
            raise ValueError(f"Invalid frequency format: {freq_str}")

        numeric_part, unit = match.groups()
        freq_mhz = float(numeric_part) * 1000 if unit == "ghz" else float(numeric_part)
        return freq_mhz
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
    pred = model(input_arr_scaled, training=False)  # Faster inference
    return pred.numpy().flatten()


def inverse_predict(target, initial_guess):
    """
    Optimize input parameters to match the desired target output.
    Validates optimizer success and ensures bounds are respected.
    """

    def objective(x):
        pred = forward_predict(x)
        return np.sum((pred - target) ** 2)

    # Unified bounds (consistent with main function)
    bounds = [
        (7, 13), (6, 25), (6, 25),  # l_s, l_2, l_1
        (0.20, 0.6), (0.20, 0.6),  # s_2, s_1
        (0.6, 1.8), (0.5, 2), (0.6, 2),  # w_s, w_2, w_1
        (800, 4000)  # freq (MHz)
    ]

    # Clamping initial guess within bounds
    initial_guess = np.clip(initial_guess, [b[0] for b in bounds], [b[1] for b in bounds])

    res = minimize(objective, x0=initial_guess, bounds=bounds, method='L-BFGS-B')

    if not res.success:
        print("\n⚠️ Warning: Optimization may not have converged. Results might be suboptimal.")

    return res.x, res.fun


def main():
    print("Welcome to the Advanced Prediction Script!")
    print("Choose a mode:")
    print("1. Forward Prediction: Provide input parameters to get the model's output.")
    print("2. Inverse Prediction: Provide desired target output values to find input parameters that produce them.")
    mode = input("Enter mode (forward/inverse): ").strip().lower()

    if mode == "forward":
        print("\nForward Prediction Mode:")
        print("Enter 9 comma-separated values [l_s, l_2, l_1, s_2, s_1, w_s, w_2, w_1, freq]")
        print("Example: 6.0,7.0,7.0,0.15,0.15,0.6,0.5,0.2,3.600 GHz")

        input_str = input("Input: ")
        try:
            input_params = parse_forward_input(input_str)
            predicted = forward_predict(input_params)
            print(f"Model predicted output: {predicted}")
        except ValueError as e:
            print(f"Error: {e}")

    elif mode == "inverse":
        print("\nInverse Prediction Mode:")
        print("Enter desired dB values for [dB(S(1,1)), dB(S(2,1))] (comma-separated)")
        print("Example: -2.528,-3.598")

        target_str = input("Target: ")
        try:
            target = np.array([float(val) for val in target_str.split(",")])
            if target.shape[0] != 2:
                raise ValueError("Exactly 2 target values are required.")
        except ValueError as e:
            print(f"Error: {e}")
            return

        print("Enter an initial guess for 9 input parameters or press Enter for default.")
        print("Example: 6.0,7.0,7.0,0.15,0.15,0.6,0.5,0.2,3.300 GHz")

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

        solution, error_val = inverse_predict(target, initial_guess)

        formatted_values = [f"{val:.3f}" for val in solution[:-1]]
        freq_display = f"{solution[-1] / 1000:.3f} GHz" if solution[-1] >= 1000 else f"{solution[-1]:.3f} MHz"
        formatted_solution = ", ".join(formatted_values + [freq_display])

        print("\nOptimized Input Parameters:")
        print(formatted_solution)
        print(f"Final objective (error value): {error_val:.6f}")

    else:
        print("Invalid mode. Please restart and choose 'forward' or 'inverse'.")


if __name__ == "__main__":
    main()
