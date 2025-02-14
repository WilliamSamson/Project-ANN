import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.optimize import minimize
import joblib
import re
import matplotlib.pyplot as plt

# Load Model & Scaler
model = load_model('/home/kayode-olalere/PycharmProjects/Project ANN/Codebase/Model/VER_3/v_1/best_model.h5',
                   compile=False)
scaler = joblib.load('/home/kayode-olalere/PycharmProjects/Project ANN/Codebase/Model/VER_3/v_1/scaler.pkl')


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


def forward_predict(input_params):
    """Scale input and predict output using the model."""
    # Ensure input has exactly 9 features
    if len(input_params) != 9:
        raise ValueError(f"Expected 9 features, got {len(input_params)}")
    input_arr = np.array(input_params, dtype='float32').reshape(1, -1)
    input_arr_scaled = scaler.transform(input_arr)
    pred = model(input_arr_scaled, training=False)
    return pred.numpy().flatten()


def calculate_accuracy(actual, predicted):
    """Compute accuracy using a normalized error approach (for inverse mode only)."""
    actual = np.array(actual, dtype=np.float32)
    predicted = np.array(predicted, dtype=np.float32)
    # Relative error calculation; note that if error > 100%, accuracy becomes negative.
    denominator = np.maximum(np.maximum(np.abs(actual), np.abs(predicted)), 1e-6)
    error = np.abs((actual - predicted) / denominator)
    mean_error = np.mean(error)
    accuracy_percentage = 100 * (1 - mean_error)
    return accuracy_percentage


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


def dual_frequency_optimization(initial_params, freq1, freq2):
    """Optimize design parameters (8 numbers) for dual-frequency performance."""
    # Target: dB(S(1,1)) = -10, dB(S(2,1)) = -1 at both frequencies.
    target_s1 = -10
    target_s2 = -1
    target = np.array([target_s1, target_s2, target_s1, target_s2], dtype=np.float32)

    def objective(x):
        # Build full input arrays (8 parameters + frequency)
        input_f1 = np.append(x, freq1).astype('float32').reshape(1, -1)
        input_f2 = np.append(x, freq2).astype('float32').reshape(1, -1)
        pred_f1 = model(scaler.transform(input_f1), training=False).numpy().flatten()
        pred_f2 = model(scaler.transform(input_f2), training=False).numpy().flatten()
        combined_pred = np.concatenate([pred_f1, pred_f2])
        return np.sum((combined_pred - target) ** 2)

    bounds = [
        (7, 13), (6, 25), (6, 25),
        (0.20, 0.6), (0.20, 0.6),
        (0.6, 1.8), (0.5, 2), (0.6, 2)
    ]
    res = minimize(objective, x0=initial_params, bounds=bounds, method='L-BFGS-B')
    return res.x, res.fun


def plot_performance(initial_params, optimized_params, freq1, freq2):
    """Generate plots for frequency response (S-parameters) vs frequency."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Parameter Comparison (bar chart)
    param_names = ['l_s', 'l_2', 'l_1', 's_2', 's_1', 'w_s', 'w_2', 'w_1']
    axs[0, 0].bar(np.arange(len(param_names)) - 0.15, initial_params, width=0.3, label='Initial')
    axs[0, 0].bar(np.arange(len(param_names)) + 0.15, optimized_params, width=0.3, label='Optimized')
    axs[0, 0].set_title('Design Parameter Comparison')
    axs[0, 0].set_xticks(range(len(param_names)))
    axs[0, 0].set_xticklabels(param_names, rotation=45)
    axs[0, 0].legend()

    # 2. Frequency Response Sweep (Predicted S-parameters vs frequency)
    freqs = np.linspace(800, 4000, 50)
    predictions = [forward_predict(np.append(np.array(optimized_params), f)) for f in freqs]
    s11 = [p[0] for p in predictions]
    s21 = [p[1] for p in predictions]
    axs[0, 1].plot(freqs, s11, label='dB(S(1,1))')
    axs[0, 1].plot(freqs, s21, label='dB(S(2,1))')
    axs[0, 1].axvline(freq1, color='r', linestyle='--', label=f'Freq1: {freq1} MHz')
    axs[0, 1].axvline(freq2, color='g', linestyle='--', label=f'Freq2: {freq2} MHz')
    axs[0, 1].set_title('Frequency Response')
    axs[0, 1].set_xlabel('Frequency (MHz)')
    axs[0, 1].set_ylabel('dB')
    axs[0, 1].legend()

    # 3. Target Achievement: Compare predicted S-parameters for initial and optimized sets at each target frequency
    initial_pred_f1 = forward_predict(np.append(np.array(initial_params), freq1))
    initial_pred_f2 = forward_predict(np.append(np.array(initial_params), freq2))
    opt_pred_f1 = forward_predict(list(optimized_params) + [freq1])
    opt_pred_f2 = forward_predict(list(optimized_params) + [freq2])
    axs[1, 0].bar(['Freq1 S11', 'Freq1 S21'], [initial_pred_f1[0], initial_pred_f1[1]],
                  width=0.4, label='Initial', alpha=0.6)
    axs[1, 0].bar(['Freq1 S11', 'Freq1 S21'], [opt_pred_f1[0], opt_pred_f1[1]],
                  width=0.4, label='Optimized', alpha=0.6)
    axs[1, 0].axhline(-10, color='r', linestyle='--', label='S11 Target')
    axs[1, 0].axhline(-1, color='g', linestyle='--', label='S21 Target')
    axs[1, 0].set_title('Freq1 S-parameter Comparison')
    axs[1, 0].legend()

    axs[1, 1].bar(['Freq2 S11', 'Freq2 S21'], [initial_pred_f2[0], initial_pred_f2[1]],
                  width=0.4, label='Initial', alpha=0.6)
    axs[1, 1].bar(['Freq2 S11', 'Freq2 S21'], [opt_pred_f2[0], opt_pred_f2[1]],
                  width=0.4, label='Optimized', alpha=0.6)
    axs[1, 1].axhline(-10, color='r', linestyle='--', label='S11 Target')
    axs[1, 1].axhline(-1, color='g', linestyle='--', label='S21 Target')
    axs[1, 1].set_title('Freq2 S-parameter Comparison')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig('dual_frequency_analysis.png')
    plt.close()


def plot_design_parameters(initial_params, optimized_params):
    """Plot design parameters vs. frequency (they are constant, so we show horizontal lines)"""
    freqs = np.linspace(800, 4000, 50)

    # Plot Lengths (first three parameters)
    plt.figure(figsize=(10, 6))
    for i, name in enumerate(['l_s', 'l_2', 'l_1']):
        plt.plot(freqs, [initial_params[i]] * len(freqs), '--', label=f'Initial {name}')
        plt.plot(freqs, [optimized_params[i]] * len(freqs), '-', label=f'Optimized {name}')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Length')
    plt.title('Frequency vs. Length')
    plt.legend()
    plt.savefig('frequency_vs_length.png')
    plt.close()

    # Plot Spacings (parameters 4 and 5: s_2, s_1)
    plt.figure(figsize=(10, 6))
    for i, name in zip([3, 4], ['s_2', 's_1']):
        plt.plot(freqs, [initial_params[i]] * len(freqs), '--', label=f'Initial {name}')
        plt.plot(freqs, [optimized_params[i]] * len(freqs), '-', label=f'Optimized {name}')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Spacing')
    plt.title('Frequency vs. Spacing')
    plt.legend()
    plt.savefig('frequency_vs_spacing.png')
    plt.close()

    # Plot Widths (parameters 6,7,8: w_s, w_2, w_1)
    plt.figure(figsize=(10, 6))
    for i, name in zip([5, 6, 7], ['w_s', 'w_2', 'w_1']):
        plt.plot(freqs, [initial_params[i]] * len(freqs), '--', label=f'Initial {name}')
        plt.plot(freqs, [optimized_params[i]] * len(freqs), '-', label=f'Optimized {name}')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Width')
    plt.title('Frequency vs. Width')
    plt.legend()
    plt.savefig('frequency_vs_width.png')
    plt.close()


def main():
    print("Welcome to the Enhanced Prediction System!")
    print("Choose a mode:")
    print("1. Forward Prediction")
    print("2. Inverse Prediction")
    print("3. Dual-Frequency Optimization")
    mode = input("Enter mode (forward/inverse/dual): ").strip().lower()

    if mode == "dual":
        print("\nDual-Frequency Optimization Mode:")
        print("Enter 8 base parameters [l_s, l_2, l_1, s_2, s_1, w_s, w_2, w_1]")
        print("Followed by two frequencies (comma-separated)")
        input_str = input("Input (10 values total): ")
        try:
            parts = input_str.split(",")
            if len(parts) != 10:
                raise ValueError("Exactly 10 values required (8 params + 2 freqs)")
            base_params = [float(p) for p in parts[:8]]
            freq1 = parse_frequency(parts[8])
            freq2 = parse_frequency(parts[9])

            # Compute initial predictions
            init_pred_f1 = forward_predict(base_params + [freq1])
            init_pred_f2 = forward_predict(base_params + [freq2])

            print("\n--- Initial Predictions ---")
            print(f"At Frequency {freq1} MHz: dB(S11) = {init_pred_f1[0]:.2f}, dB(S21) = {init_pred_f1[1]:.2f}")
            print(f"At Frequency {freq2} MHz: dB(S11) = {init_pred_f2[0]:.2f}, dB(S21) = {init_pred_f2[1]:.2f}")
            print(f"Initial Design Parameters: {base_params}")

            # Check if initial predictions are within thresholds
            thresholds = {'s11': 1.0, 's21': 0.2}  # Allowable deviation
            within_range = (
                    (abs(init_pred_f1[0] + 10) < thresholds['s11']) and
                    (abs(init_pred_f1[1] + 1) < thresholds['s21']) and
                    (abs(init_pred_f2[0] + 10) < thresholds['s11']) and
                    (abs(init_pred_f2[1] + 1) < thresholds['s21'])
            )
            if within_range:
                print("Initial parameters already meet targets!")
                optimized_params = base_params
                final_error = 0.0
            else:
                print("Optimizing parameters to meet targets...")
                optimized_params, final_error = dual_frequency_optimization(base_params, freq1, freq2)

            # Compute final predictions
            final_pred_f1 = forward_predict(list(optimized_params) + [freq1])
            final_pred_f2 = forward_predict(list(optimized_params) + [freq2])

            print("\n--- Final Predictions ---")
            print(f"At Frequency {freq1} MHz: dB(S11) = {final_pred_f1[0]:.2f}, dB(S21) = {final_pred_f1[1]:.2f}")
            print(f"At Frequency {freq2} MHz: dB(S11) = {final_pred_f2[0]:.2f}, dB(S21) = {final_pred_f2[1]:.2f}")
            print(f"Optimized Design Parameters: {list(optimized_params)}")

            # Calculate accuracy (note: if relative error > 100%, accuracy becomes negative)
            def calc_accuracy(preds, targets):
                errors = np.abs((preds - targets) / np.maximum(np.abs(targets), 1e-6))
                return 100 * (1 - np.mean(errors))

            accuracy = calc_accuracy(np.concatenate([final_pred_f1, final_pred_f2]), [-10, -1, -10, -1])

            # Generate performance plots and additional design parameter plots
            plot_performance(base_params, optimized_params, freq1, freq2)
            plot_design_parameters(base_params, list(optimized_params))

            print("\nOptimization Complete!")
            print(f"Final Accuracy: {accuracy:.2f}%")
            print(f"Error Value: {final_error:.4f}")
            print("Visual analysis saved to 'dual_frequency_analysis.png', 'frequency_vs_length.png',")
            print(" 'frequency_vs_spacing.png', and 'frequency_vs_width.png'")

        except ValueError as e:
            print(f"Error: {e}")

    elif mode == "forward":
        print("\nForward Prediction Mode:")
        print("Enter 9 comma-separated values [l_s, l_2, l_1, s_2, s_1, w_s, w_2, w_1, freq]")
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
        solution, error_val, accuracy_percentage = inverse_predict(target, initial_guess)
        formatted_values = [f"{val:.3f}" for val in solution[:-1]]
        freq_display = f"{solution[-1] / 1000:.3f} GHz" if solution[-1] >= 1000 else f"{solution[-1]:.3f} MHz"
        formatted_solution = ", ".join(formatted_values + [freq_display])
        print("\nOptimized Input Parameters:")
        print(formatted_solution)
        print(f"Final objective (error value): {error_val:.6f}")
        print(f"Accuracy: {accuracy_percentage:.2f}%")
    else:
        print("Invalid mode. Please restart and choose 'forward', 'inverse', or 'dual'.")


if __name__ == "__main__":
    main()
