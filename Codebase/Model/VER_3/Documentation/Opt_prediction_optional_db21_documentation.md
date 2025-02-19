
---

# Comprehensive Documentation for the Dual-Frequency Antenna Optimization and Prediction Script

## 1. Introduction

This script is a versatile tool for antenna design and optimization. It provides three modes of operation:

- **Forward Prediction:**
  Given a set of input design parameters (eight design features plus one frequency), the script uses a pre-trained neural network model to predict the antenna’s S-parameters (e.g., dB(S(1,1)) and, optionally, dB(S(2,1))).

- **Inverse Prediction:**
  Given a desired target S-parameter output, the script optimizes the input parameters so that the model’s predictions are as close as possible to the target. This “inverts” the forward mapping.

- **Dual-Frequency Optimization:**
  Simultaneously optimizes the design parameters to meet performance targets at two distinct frequencies. In this mode, the script employs a two-tier optimization approach:
  1. **Global Optimization:** Uses differential evolution (DE) to broadly search the design space.
  2. **Local Optimization:** Uses the L‑BFGS‑B algorithm for fine-tuning the candidate obtained from DE.

  In addition, the script concurrently tests several bounds configurations (using `concurrent.futures`) to determine a data‐driven “best” bound set based on the global error.

The code leverages TensorFlow for the neural network model, SciPy for optimization, and Matplotlib for visualizations. It also uses Joblib to load the scaler, ensuring that new inputs are processed in the same way as the training data. For interactive command-line prompts, it uses Questionary.

---

## 2. Code Overview

### 2.1 Imports and Setup


```
python
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.optimize import minimize, differential_evolution
import joblib
import re
import matplotlib.pyplot as plt
from pathlib import Path
import questionary
import concurrent.futures
```


**Why?**
- **Libraries:**
  - **TensorFlow/Keras:** Loads and runs the pre-trained model.
  - **SciPy:** Provides optimization routines (DE and L‑BFGS‑B) to invert the forward model.
  - **Joblib:** Loads the scaler used to preprocess input data.
  - **Matplotlib:** Generates visual output (plots) for performance and design parameter comparisons.
  - **Questionary:** Supplies interactive CLI prompts with arrow-key selection for a more user-friendly experience.
  - **Concurrent.futures:** Runs bounds-testing evaluations in parallel to speed up the selection process.

- **Graph Directory Setup:**
  A folder named `Graphs` is created (if it does not exist) to store all generated visual outputs.

### 2.2 Model and Scaler Loading


```
python
project_root = Path(__file__).resolve().parents[1]
data_path = project_root / "best_model.h5"
data_path2 = project_root / "scaler.pkl"

if not data_path.exists():
    raise FileNotFoundError(f"File not found: {data_path}")

if not data_path2.exists():
    raise FileNotFoundError(f"File not found: {data_path2}")

model = load_model(str(data_path), compile=False)
scaler = joblib.load(str(data_path2))
```


**Why?**
- **Dynamic Paths:** Using `Path` ensures the code is portable across different systems.
- **Error Handling:** The code checks for the existence of the model and scaler files, aborting execution if they are missing.
- **Loading:** The trained model and the scaler (used to normalize input features) are loaded so that inference is performed exactly as during training.

### 2.3 Global Variable


```
python
INCLUDE_S21 = True
```


**Purpose:**
Controls whether the script will consider both S-parameters (S11 and S21) or only S11 in predictions and optimizations. This flag is later set interactively via Questionary.

---

## 3. Input Parsing and Prediction Functions

### 3.1 `parse_forward_input(input_str)`


```
python
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
```


**Why?**
- **Validation:** Ensures the user provides exactly 9 inputs (8 design parameters + frequency).
- **Standardization:** Converts the frequency string (which can include “MHz” or “GHz”) to a numeric value in MHz using the helper function `parse_frequency`.

### 3.2 `parse_frequency(freq_str)`


```
python
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
```


**Why?**
- **Regex Matching:** Extracts both the numeric part and the unit.
- **Unit Conversion:** Converts GHz values to MHz for consistency.

### 3.3 `forward_predict(input_params)`


```
python
def forward_predict(input_params):
    """Scale input and predict output using the model."""
    if len(input_params) != 9:
        raise ValueError(f"Expected 9 features, got {len(input_params)}")
    input_arr = np.array(input_params, dtype='float32').reshape(1, -1)
    input_arr_scaled = scaler.transform(input_arr)
    pred = model(input_arr_scaled, training=False)
    prediction = pred.numpy().flatten()
    if not INCLUDE_S21:
        return prediction[:1]  # Only return S11 if S21 is not used
    return prediction
```


**Why?**
- **Input Scaling:** Ensures the input features are normalized in the same way as during training.
- **Model Inference:** Performs inference in non-training mode (to disable dropout, etc.).
- **Optional Output:** Depending on `INCLUDE_S21`, returns either both predicted S-parameters or just S11.

### 3.4 `calculate_accuracy(actual, predicted)`


```
python
def calculate_accuracy(actual, predicted):
    """Compute accuracy using a normalized error approach."""
    actual = np.array(actual, dtype=np.float32)
    predicted = np.array(predicted, dtype=np.float32)
    denominator = np.maximum(np.maximum(np.abs(actual), np.abs(predicted)), 1e-6)
    error = np.abs((actual - predicted) / denominator)
    mean_error = np.mean(error)
    accuracy_percentage = 100 * (1 - mean_error)
    return accuracy_percentage
```


**Why?**
- **Normalization:** Avoids division by zero.
- **Interpretability:** Converts error into an accuracy percentage for easier interpretation.

---

## 4. Optimization Functions

### 4.1 Inverse Prediction


```
python
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
```


**Why?**
- **Objective Function:** Minimizes the squared error between predicted and target S-parameters.
- **Bounds & Clipping:** Ensures solutions remain within practical limits.
- **Optimization:** Uses the efficient L-BFGS-B algorithm for local search.

### 4.2 Dual-Frequency Optimization

#### 4.2.1 Dual Objective Function


```
python
def dual_objective(x, freq1, freq2):
    if INCLUDE_S21:
        target = np.array([-15, -1, -15, -1], dtype=np.float32)
    else:
        target = np.array([-15, -15], dtype=np.float32)

    # Append each frequency to the design parameters
    input_f1 = np.append(x, freq1).astype('float32').reshape(1, -1)
    input_f2 = np.append(x, freq2).astype('float32').reshape(1, -1)

    # Flatten the reshaped array as forward_predict expects a 9-element vector
    pred_f1 = forward_predict(input_f1.flatten())
    pred_f2 = forward_predict(input_f2.flatten())

    combined_pred = np.concatenate([pred_f1, pred_f2])
    return np.sum((combined_pred - target) ** 2)
```


**Why?**
- **Dual Targets:** The function computes error across two frequencies by concatenating predictions.
- **Frequency Appending:** For each candidate design (x), the specified frequency is appended before passing it to the model.
- **Target Adjustments:** The target for S11 is set to –15 dB (and –1 dB for S21 if included).

#### 4.2.2 Global Optimization


```
python
def global_dual_frequency_optimization(freq1, freq2, bounds):
    """
    Global optimization using differential evolution to search broadly for the design parameters.
    """
    result = differential_evolution(dual_objective, bounds=bounds, args=(freq1, freq2),
                                    strategy='best1bin', maxiter=100, popsize=15,
                                    tol=0.01, mutation=(0.5, 1), recombination=0.7)
    return result.x, result.fun
```


**Why?**
- **Differential Evolution:** Explores a wide search space to find promising candidate designs.
- **Robust Global Search:** Helps avoid local minima and provide a good initial candidate for further refinement.

#### 4.2.3 Local Optimization


```
python
def local_dual_frequency_optimization(initial_guess, freq1, freq2, bounds):
    """
    Local optimization using L-BFGS-B for fine-tuning starting from an initial candidate.
    """
    res = minimize(dual_objective, x0=initial_guess, args=(freq1, freq2), bounds=bounds, method='L-BFGS-B')
    return res.x, res.fun
```


**Why?**
- **Refinement:** Uses L-BFGS-B to fine-tune the candidate from the global search, ensuring better precision near the optimum.

---

## 5. Concurrent Bounds Testing

### 5.1 Helper Function: `evaluate_bounds`


```
python
def evaluate_bounds(b, freq1, freq2):
    """
    Evaluate a single bounds configuration.
    Returns a tuple containing:
      - the bounds configuration,
      - the global candidate,
      - the global error,
      - the local candidate,
      - the local error,
      - final predictions at freq1,
      - final predictions at freq2.
    """
    global_candidate, global_error = global_dual_frequency_optimization(freq1, freq2, b)
    local_candidate, local_error = local_dual_frequency_optimization(global_candidate, freq1, freq2, b)
    final_pred_f1 = forward_predict(list(local_candidate) + [freq1])
    final_pred_f2 = forward_predict(list(local_candidate) + [freq2])
    return (b, global_candidate, global_error, local_candidate, local_error, final_pred_f1, final_pred_f2)
```


**Why?**
- **Modular Evaluation:** This function encapsulates the entire optimization process (global and local) for one set of bounds.
- **Output Tuple:** It returns detailed results, which can later be compared to select the best bounds configuration.

### 5.2 Helper Function: `run_possible_bounds`


```
python
def run_possible_bounds(freq1, freq2):
    """
    Tests several bounds configurations concurrently and returns the best bounds (i.e., with the lowest global error)
    along with its corresponding local candidate.
    """
    possible_bounds = [
        [(6, 13), (7, 25), (7, 25),
         (0.15, 0.6), (0.15, 0.6),
         (0.6, 1.8), (0.5, 2.0), (0.2, 2.0)],
        [(7, 13), (6, 25), (6, 25),
         (0.20, 0.6), (0.20, 0.6),
         (0.6, 1.8), (0.5, 2.0), (0.6, 2.0)],
        # Additional bound configurations can be added here.
    ]

    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(evaluate_bounds, b, freq1, freq2): b for b in possible_bounds}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            b_config, global_candidate, global_error, local_candidate, local_error, final_pred_f1, final_pred_f2 = result
            print("Testing bounds:", b_config)
            print("Global candidate:", format_design_parameters(global_candidate))
            print(f"Global error: {global_error:.4f}")
            if INCLUDE_S21:
                print(f"Final Predictions at {freq1} MHz: S11 = {final_pred_f1[0]:.2f}, S21 = {final_pred_f1[1]:.2f}")
                print(f"Final Predictions at {freq2} MHz: S11 = {final_pred_f2[0]:.2f}, S21 = {final_pred_f2[1]:.2f}")
            else:
                print(f"Final Predictions at {freq1} MHz: S11 = {final_pred_f1[0]:.2f}")
                print(f"Final Predictions at {freq2} MHz: S11 = {final_pred_f2[0]:.2f}")
            print("-" * 50)

    # Choose the best candidate based on the lowest global error
    best = min(results, key=lambda x: x[2])
    best_bounds, best_global_candidate, best_global_error, best_local_candidate, best_local_error, best_final_pred_f1, best_final_pred_f2 = best
    print("Best bounds configuration selected:")
    print("Bounds:", best_bounds)
    print("Candidate:", format_design_parameters(best_local_candidate))
    print("Global error:", best_global_error)
    return best_bounds, best_local_candidate
```


**Why?**
- **Concurrent Execution:** Uses `concurrent.futures.ThreadPoolExecutor` to evaluate all bounds configurations in parallel, which speeds up the testing process.
- **Selection:** Compares the results from each configuration (based on global error) and selects the best one.
- **Output:** Returns the best bounds and the corresponding candidate, which will be used as the final optimized design.

---

## 6. Visualization Functions

### 6.1 `format_design_parameters`


```
python
def format_design_parameters(params):
    """Return a friendly formatted string of design parameters."""
    names = ['l_s', 'l_2', 'l_1', 's_2', 's_1', 'w_s', 'w_2', 'w_1']
    return ", ".join([f"{name}: {float(val):.3f}" for name, val in zip(names, params)])
```


**Why?**
- **Readability:** Converts the list of design parameters into a human-friendly string for logging and reporting.

### 6.2 `plot_performance`


```
python
def plot_performance(initial_params, optimized_params, freq1, freq2):
    """Generate plots for frequency response vs frequency and S-parameter comparisons."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # Parameter Comparison Bar Chart
    param_names = ['l_s', 'l_2', 'l_1', 's_2', 's_1', 'w_s', 'w_2', 'w_1']
    axs[0, 0].bar(np.arange(len(param_names)) - 0.15, initial_params, width=0.3, label='Initial')
    axs[0, 0].bar(np.arange(len(param_names)) + 0.15, optimized_params, width=0.3, label='Optimized')
    axs[0, 0].set_title('Design Parameter Comparison')
    axs[0, 0].set_xticks(range(len(param_names)))
    axs[0, 0].set_xticklabels(param_names, rotation=45)
    axs[0, 0].legend()

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

    # S-parameter Comparison at Freq1
    initial_pred_f1 = forward_predict(np.append(np.array(initial_params), freq1))
    opt_pred_f1 = forward_predict(list(optimized_params) + [freq1])
    if INCLUDE_S21:
        axs[1, 0].bar(['Freq1 S11', 'Freq1 S21'], [initial_pred_f1[0], initial_pred_f1[1]],
                      width=0.4, label='Initial', alpha=0.6)
        axs[1, 0].bar(['Freq1 S11', 'Freq1 S21'], [opt_pred_f1[0], opt_pred_f1[1]],
                      width=0.4, label='Optimized', alpha=0.6)
        axs[1, 0].axhline(-15, color='r', linestyle='--', label='S11 Target')
        axs[1, 0].axhline(-1, color='g', linestyle='--', label='S21 Target')
    else:
        axs[1, 0].bar(['Freq1 S11'], [initial_pred_f1[0]],
                      width=0.4, label='Initial', alpha=0.6)
        axs[1, 0].bar(['Freq1 S11'], [opt_pred_f1[0]],
                      width=0.4, label='Optimized', alpha=0.6)
        axs[1, 0].axhline(-15, color='r', linestyle='--', label='S11 Target')
    axs[1, 0].set_title('Freq1 S-parameter Comparison')
    axs[1, 0].legend()

    # S-parameter Comparison at Freq2
    initial_pred_f2 = forward_predict(np.append(np.array(initial_params), freq2))
    opt_pred_f2 = forward_predict(list(optimized_params) + [freq2])
    if INCLUDE_S21:
        axs[1, 1].bar(['Freq2 S11', 'Freq2 S21'], [initial_pred_f2[0], initial_pred_f2[1]],
                      width=0.4, label='Initial', alpha=0.6)
        axs[1, 1].bar(['Freq2 S11', 'Freq2 S21'], [opt_pred_f2[0], opt_pred_f2[1]],
                      width=0.4, label='Optimized', alpha=0.6)
        axs[1, 1].axhline(-15, color='r', linestyle='--', label='S11 Target')
        axs[1, 1].axhline(-1, color='g', linestyle='--', label='S21 Target')
    else:
        axs[1, 1].bar(['Freq2 S11'], [initial_pred_f2[0]],
                      width=0.4, label='Initial', alpha=0.6)
        axs[1, 1].bar(['Freq2 S11'], [opt_pred_f2[0]],
                      width=0.4, label='Optimized', alpha=0.6)
        axs[1, 1].axhline(-15, color='r', linestyle='--', label='S11 Target')
    axs[1, 1].set_title('Freq2 S-parameter Comparison')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_SAVE_DIR, 'dual_frequency_analysis.png'))
    plt.close()
```


**Why?**
- **Multiple Views:**
  - The first subplot compares initial versus optimized design parameters.
  - The second subplot shows the frequency response of the optimized design.
  - The third and fourth subplots display the S-parameter comparisons at the two frequencies.
- **Target Lines:** Horizontal lines (e.g., at –15 dB for S11) visually indicate the desired performance.
- **File Organization:** The resulting plots are saved in the designated `Graphs` folder.

### 6.3 `plot_design_parameters`


```
python
def plot_design_parameters(initial_params, optimized_params):
    """Plot design parameters vs. frequency (displayed as constant horizontal lines)."""
    freqs = np.linspace(800, 4000, 50)
    # Plot Lengths (l_s, l_2, l_1)
    plt.figure(figsize=(10, 6))
    for i, name in enumerate(['l_s', 'l_2', 'l_1']):
        plt.plot(freqs, [initial_params[i]] * len(freqs), '--', label=f'Initial {name}')
        plt.plot(freqs, [optimized_params[i]] * len(freqs), '-', label=f'Optimized {name}')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Length')
    plt.title('Frequency vs. Length')
    plt.legend()
    plt.savefig(os.path.join(IMAGE_SAVE_DIR, 'frequency_vs_length.png'))
    plt.close()

    # Plot Spacings (s_2, s_1)
    plt.figure(figsize=(10, 6))
    for i, name in zip([3, 4], ['s_2', 's_1']):
        plt.plot(freqs, [initial_params[i]] * len(freqs), '--', label=f'Initial {name}')
        plt.plot(freqs, [optimized_params[i]] * len(freqs), '-', label=f'Optimized {name}')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Spacing')
    plt.title('Frequency vs. Spacing')
    plt.legend()
    plt.savefig(os.path.join(IMAGE_SAVE_DIR, 'frequency_vs_spacing.png'))
    plt.close()

    # Plot Widths (w_s, w_2, w_1)
    plt.figure(figsize=(10, 6))
    for i, name in zip([5, 6, 7], ['w_s', 'w_2', 'w_1']):
        plt.plot(freqs, [initial_params[i]] * len(freqs), '--', label=f'Initial {name}')
        plt.plot(freqs, [optimized_params[i]] * len(freqs), '-', label=f'Optimized {name}')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Width')
    plt.title('Frequency vs. Width')
    plt.legend()
    plt.savefig(os.path.join(IMAGE_SAVE_DIR, 'frequency_vs_width.png'))
    plt.close()
```


**Why?**
- **Parameter Trends:** These functions plot each design parameter as a constant (horizontal line) over the frequency range, allowing a visual comparison of initial versus optimized values.
- **Organization:** The plots are saved in the `Graphs` directory for later review.

---

## 7. Main Function and User Interaction


```
python
def main():
    global INCLUDE_S21

    # Interactive selection for S21 inclusion
    include_choice = questionary.select(
        "Include dB S21 values in predictions?",
        choices=["Yes", "No"]
    ).ask()
    INCLUDE_S21 = True if include_choice == "Yes" else False

    # Interactive selection of mode
    mode = questionary.select(
        "Select a mode:",
        choices=["Forward Prediction", "Inverse Prediction", "Dual-Frequency Optimization"]
    ).ask()

    if mode == "Dual-Frequency Optimization":
        print("\nDual-Frequency Optimization Mode:")
        print("\nEnter 10 comma-separated values: 8 base parameters [l_s, l_2, l_1, s_2, s_1, w_s, w_2, w_1] followed by two frequencies:")
        prompt_text = "Input:"
        input_str = questionary.text(prompt_text).ask()
        parts = input_str.split(",")
        if len(parts) != 10:
            raise ValueError("Exactly 10 values required (8 params + 2 freqs)")
        base_params = [float(p) for p in parts[:8]]
        freq1 = parse_frequency(parts[8])
        freq2 = parse_frequency(parts[9])

        # Display initial predictions based on the provided base parameters.
        init_pred_f1 = forward_predict(base_params + [freq1])
        init_pred_f2 = forward_predict(base_params + [freq2])
        print("\n--- Initial Predictions ---")
        if INCLUDE_S21:
            print(f"At {freq1} MHz: S11 = {init_pred_f1[0]:.2f}, S21 = {init_pred_f1[1]:.2f}")
            print(f"At {freq2} MHz: S11 = {init_pred_f2[0]:.2f}, S21 = {init_pred_f2[1]:.2f}")
        else:
            print(f"At {freq1} MHz: S11 = {init_pred_f1[0]:.2f}")
            print(f"At {freq2} MHz: S11 = {init_pred_f2[0]:.2f}")
        print("Initial Design Parameters:")
        print(format_design_parameters(base_params))

        # Run concurrent bounds testing and select the best configuration.
        best_bounds, best_candidate = run_possible_bounds(freq1, freq2)

        # Use the best candidate for the final optimized output.
        final_pred_f1 = forward_predict(list(best_candidate) + [freq1])
        final_pred_f2 = forward_predict(list(best_candidate) + [freq2])
        print("\n--- Final Predictions (using best bounds configuration) ---")
        if INCLUDE_S21:
            print(f"At {freq1} MHz: S11 = {final_pred_f1[0]:.2f}, S21 = {final_pred_f1[1]:.2f}")
            print(f"At {freq2} MHz: S11 = {final_pred_f2[0]:.2f}, S21 = {final_pred_f2[1]:.2f}")
        else:
            print(f"At {freq1} MHz: S11 = {final_pred_f1[0]:.2f}")
            print(f"At {freq2} MHz: S11 = {final_pred_f2[0]:.2f}")
        print("Optimized Design Parameters:")
        print(format_design_parameters(best_candidate))

        # Generate performance and design parameter plots.
        plot_performance(base_params, best_candidate, freq1, freq2)
        plot_design_parameters(base_params, list(best_candidate))

    elif mode == "Forward Prediction":
        print("\nForward Prediction Mode:")
        print("Enter 9 comma-separated values [l_s, l_2, l_1, s_2, s_1, w_s, w_2, w_1, freq]")
        prompt_text = "Input:"
        input_str = questionary.text(prompt_text).ask()
        try:
            input_params = parse_forward_input(input_str)
            predicted = forward_predict(input_params)
            if INCLUDE_S21:
                print(f"Model predicted output: {predicted}")
            else:
                print(f"Model predicted S11: {predicted[0]:.2f}")
        except ValueError as e:
            print(f"Error: {e}")

    elif mode == "Inverse Prediction":
        print("\nInverse Prediction Mode:")
        if INCLUDE_S21:
            target_prompt = "Enter desired dB values for [dB(S(1,1)), dB(S(2,1))] (comma-separated)"
        else:
            target_prompt = "Enter desired dB value for [dB(S(1,1))]"
        target_str = questionary.text(target_prompt).ask()
        try:
            if INCLUDE_S21:
                target_vals = [float(val) for val in target_str.split(",")]
                if len(target_vals) != 2:
                    raise ValueError("Exactly 2 target values are required.")
                target = np.array(target_vals)
            else:
                target = np.array([float(target_str)])
        except ValueError as e:
            print(f"Error: {e}")
            return

        print("Enter an initial guess for 9 input parameters or leave blank for default.")
        guess_str = questionary.text("Initial guess (comma-separated, 9 values):").ask().strip()
        bounds = [
            (6, 13), (6, 25), (6, 25),
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
        print("Invalid mode. Please restart and choose a valid option.")


if __name__ == "__main__":
    main()
```


**Why?**
- **Interactive Mode Selection:**
  Uses Questionary for a user-friendly, interactive CLI that allows selection among Forward Prediction, Inverse Prediction, and Dual-Frequency Optimization modes.
- **Dual-Frequency Branch:**
  In Dual-Frequency Optimization, the user provides 10 comma-separated values (8 design parameters and 2 frequencies).
  - The script first shows initial predictions based on the provided parameters.
  - Then it runs concurrent bounds testing via `run_possible_bounds()`, which evaluates several bounds configurations in parallel and selects the best candidate (lowest global error).
  - The final optimized candidate, along with final predictions, is printed and plotted.
- **Other Modes:**
  The Forward Prediction and Inverse Prediction branches use similar input parsing and error handling.

---

## 8. Final Thoughts

This advanced script demonstrates a complete pipeline for:

- Loading and pre-processing antenna design data.
- Predicting antenna S-parameters using a pre-trained deep learning model.
- Optimizing design parameters using both global (Differential Evolution) and local (L-BFGS-B) search methods.
- Concurrently testing multiple bounds configurations to select a data-driven optimum.
- Visualizing results through clear, organized plots.

The script’s modular design, robust error handling, and interactive CLI make it a flexible and powerful tool for antenna design and optimization. By integrating concurrent bounds testing, the system leverages parallel processing to efficiently explore the design space and improve performance. This comprehensive approach not only enhances accuracy but also saves time during optimization.

---

