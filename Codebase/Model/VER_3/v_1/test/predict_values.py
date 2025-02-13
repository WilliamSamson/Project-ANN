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
    """
    Convert a frequency string with an optional unit (MHz or GHz) into a numeric value in MHz.
    If no unit is provided, the number is assumed to be in MHz.
    """
    if isinstance(freq_str, str):
        freq_str = freq_str.strip()
        if "ghz" in freq_str.lower():
            numeric_part = re.sub(r'[^0-9\.]', '', freq_str)
            return float(numeric_part) * 1000
        elif "mhz" in freq_str.lower():
            numeric_part = re.sub(r'[^0-9\.]', '', freq_str)
            return float(numeric_part)
        else:
            try:
                return float(freq_str)
            except:
                return np.nan
    else:
        return float(freq_str)


def parse_forward_input(input_str):
    """
    Parse a comma-separated input string for forward prediction.
    Expects 9 values for [l_s, l_2, l_1, s_2, s_1, w_s, w_2, w_1, freq].
    For the frequency field (9th value), allows an input like "3.600 GHz".
    """
    parts = input_str.split(",")
    if len(parts) != 9:
        raise ValueError("Exactly 9 values are required.")
    # Parse the first 8 values as floats
    values = [float(val) for val in parts[:8]]
    # For frequency, use parse_frequency to handle potential unit suffixes
    freq_val = parts[8].strip()
    try:
        freq = float(freq_val)
    except ValueError:
        freq = parse_frequency(freq_val)
    values.append(freq)
    return values


def forward_predict(input_params):
    """
    Given 9 input parameters [l_s, l_2, l_1, s_2, s_1, w_s, w_2, w_1, freq] (freq in MHz),
    scale the input using the training scaler and predict the model's output.
    """
    input_arr = np.array(input_params, dtype='float32').reshape(1, -1)
    input_arr_scaled = scaler.transform(input_arr)
    pred = model.predict(input_arr_scaled)
    return pred.flatten()


def inverse_predict(target, initial_guess):
    """
    Given desired target output values (an array) and an initial guess for the 9 input parameters,
    use optimization (L-BFGS-B) to find input parameters that minimize the squared error between
    the forward prediction (with scaled input) and the target.

    Returns the optimal input parameters and the objective (error) value.
    """

    def objective(x):
        pred = forward_predict(x)
        error = np.sum((pred - target) ** 2)
        return error

    # Define bounds for each input parameter (adjust as needed)
    bounds = [
        (0, 25),  # l_s
        (0, 25),  # l_2
        (0, 25),  # l_1
        (0, 0.35),  # s_2
        (0, 0.35),  # s_1
        (0.6, 1.5),  # w_s
        (0.6, 1.5),  # w_2
        (0.6, 1.5),  # w_1
        (800, 4000)  # freq (in MHz)
    ]
    res = minimize(objective, x0=initial_guess, bounds=bounds, method='L-BFGS-B')
    return res.x, res.fun


def main():
    print("Welcome to the Advanced Prediction Script!")
    print("Choose a mode:")
    print("1. Forward Prediction: Provide input parameters to get the model's output.")
    print("2. Inverse Prediction: Provide desired target output values to find input parameters that produce them.")
    mode = input("Enter mode (forward/inverse): ").strip().lower()

    if mode == "forward":
        print("\nForward Prediction Mode:")
        print("Please enter 9 comma-separated values for [l_s, l_2, l_1, s_2, s_1, w_s, w_2, w_1, freq]")
        print("Example: 6.0,7.0,7.0,0.15,0.15,0.6,0.5,0.2,3.600 GHz")
        input_str = input("Input: ")
        try:
            input_params = parse_forward_input(input_str)
        except Exception as e:
            print(f"Error parsing input: {e}")
            return
        predicted = forward_predict(input_params)
        print(f"Model predicted output: {predicted}")

    elif mode == "inverse":
        print("\nInverse Prediction Mode:")
        print("Please enter the desired dB values for [dB(S(1,1)), dB(S(2,1))] (comma-separated, e.g., -2.528,-3.598)")
        target_str = input("Target: ")
        try:
            target = np.array([float(val) for val in target_str.split(",")])
            if target.shape[0] != 2:
                raise ValueError("Exactly 2 target values are required.")
        except Exception as e:
            print(f"Error parsing target values: {e}")
            return

        print("Now, please provide an initial guess for the 9 input parameters")
        print("Example: 6.0,7.0,7.0,0.15,0.15,0.6,0.5,0.2,3.300 GHz")
        guess_str = input("Initial guess (or press Enter to use default guess): ").strip()
        # Define bounds (same as in inverse_predict)
        bounds = [
            (7, 13),  # l_s
            (6, 25),  # l_2
            (6, 25),  # l_1
            (0.20, 0.6),  # s_2
            (0.20, 0.6),  # s_1
            (0.6, 1.8),  # w_s
            (0.5, 2),  # w_2
            (0.6, 2),  # w_1
            (800, 4000)  # freq (in MHz)
        ]
        if guess_str == "":
            # Use the midpoint of each bound as the default guess
            initial_guess = [(lb + ub) / 2 for lb, ub in bounds]
            print("Using default guess:", initial_guess)
        else:
            try:
                initial_guess = parse_forward_input(guess_str)
            except Exception as e:
                print(f"Error parsing initial guess: {e}")
                return

        solution, error_val = inverse_predict(target, initial_guess)
        print("\nOptimized Input Parameters (that should produce the desired output):")
        # Format first 8 values with three decimals
        formatted_values = [f"{val:.3f}" for val in solution[:-1]]
        # For frequency, if value >= 1000, display in GHz; otherwise, in MHz.
        if solution[-1] >= 1000:
            freq_in_ghz = solution[-1] / 1000.0
            formatted_freq = f"{freq_in_ghz:.3f} GHz"
        else:
            formatted_freq = f"{solution[-1]:.3f} MHz"
        formatted_solution = ", ".join(formatted_values + [formatted_freq])
        print(formatted_solution)
        print(f"Final objective (sum squared error): {error_val:.6f}")

    else:
        print("Invalid mode selected. Please run the script again and choose either 'forward' or 'inverse'.")


if __name__ == "__main__":
    main()
