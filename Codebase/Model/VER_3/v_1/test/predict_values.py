import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.optimize import minimize
import joblib

# Load the pre-trained model (for inference, so compile=False)
model = load_model('/home/kayode-olalere/PycharmProjects/Project ANN/Codebase/Model/VER_3/v_1/best_model.h5',
                   compile=False)

# Load the scaler that was used during training
scaler = joblib.load('/home/kayode-olalere/PycharmProjects/Project ANN/Codebase/Model/VER_3/v_1/scaler.pkl')


def forward_predict(input_params):
    """
    Given 9 input parameters [l_s, l_2, l_1, s_2, s_1, w_s, w_2, w_1, freq],
    apply the training scaler and predict the model's output.
    """
    # Ensure input is a 2D array (batch size 1)
    input_arr = np.array(input_params, dtype='float32').reshape(1, -1)
    # Scale the input as during training
    input_arr_scaled = scaler.transform(input_arr)
    pred = model.predict(input_arr_scaled)
    return pred.flatten()  # returns an array of predictions


def inverse_predict(target, initial_guess):
    """
    Given desired target output values (an array) and an initial guess for the 9 input parameters,
    use optimization to find input parameters that minimize the squared error between
    the forward prediction (with scaled input) and the target.

    Returns the optimal input parameters and the objective (error) value.
    """

    def objective(x):
        pred = forward_predict(x)  # forward_predict handles scaling internally
        error = np.sum((pred - target) ** 2)
        return error

    # Define bounds for each of the 9 input parameters.
    # Adjust these bounds based on your domain knowledge.
    bounds = [
        (0, 10),  # l_s
        (0, 10),  # l_2
        (0, 10),  # l_1
        (0, 10),  # s_2
        (0, 10),  # s_1
        (0, 10),  # w_s
        (0, 10),  # w_2
        (0, 10),  # w_1
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
        print("Example: 6.0,7.0,7.0,0.15,0.15,0.6,0.5,0.2,800")
        input_str = input("Input: ")
        try:
            input_params = [float(val) for val in input_str.split(",")]
            if len(input_params) != 9:
                raise ValueError("Exactly 9 values are required.")
        except Exception as e:
            print(f"Error parsing input: {e}")
            return

        predicted = forward_predict(input_params)
        print(f"Model predicted output: {predicted}")

    elif mode == "inverse":
        print("\nInverse Prediction Mode:")
        print("Please enter the desired dB values for [dB(S(1,1)), dB(S(2,1))] (comma-separated, e.g., -0.471,-9.909)")
        target_str = input("Target: ")
        try:
            target = np.array([float(val) for val in target_str.split(",")])
            if target.shape[0] != 2:
                raise ValueError("Exactly 2 target values are required.")
        except Exception as e:
            print(f"Error parsing target values: {e}")
            return

        print("Now, please provide an initial guess for the 9 input parameters")
        print("Example: 6.0,7.0,7.0,0.15,0.15,0.6,0.5,0.2,800")
        guess_str = input("Initial guess: ")
        try:
            initial_guess = [float(val) for val in guess_str.split(",")]
            if len(initial_guess) != 9:
                raise ValueError("Exactly 9 values are required for the initial guess.")
        except Exception as e:
            print(f"Error parsing initial guess: {e}")
            return

        solution, error_val = inverse_predict(target, initial_guess)
        print("\nOptimized Input Parameters (that should produce the desired output):")
        # Format all but the last value normally:
        formatted_values = [f"{val:.3f}" for val in solution[:-1]]
        # For the last value (frequency), add the unit:
        formatted_freq = f"{solution[-1]:.3f} MHz"
        formatted_solution = ", ".join(formatted_values + [formatted_freq])
        print(formatted_solution)
        print(f"Final objective (sum squared error): {error_val:.6f}")

    else:
        print("Invalid mode selected. Please run the script again and choose either 'forward' or 'inverse'.")


if __name__ == "__main__":
    main()
