

# Comprehensive Documentation for the Antenna Design & Optimization Script

## 1. Introduction

This script serves as a versatile tool for antenna design and optimization. It provides three modes of operation:

- **Forward Prediction:** Given a set of input design parameters, the script predicts the antenna's S-parameters (e.g., dB(S(1,1)) and dB(S(2,1))).
- **Inverse Prediction:** Given a desired target S-parameter output, the script finds the best set of input parameters to achieve that target.
- **Dual-Frequency Optimization:** Simultaneously optimizes design parameters to meet targets at two different frequencies.

The code leverages TensorFlow for the neural network model, SciPy for optimization, and Matplotlib for visualizations. It also uses Joblib to load the scaler, ensuring that new inputs are processed the same way as the training data.

---

## 2. Code Overview

### **Imports and Setup**

The script starts by importing necessary libraries for file handling, numerical computations, deep learning, optimization, and plotting. It also sets up a folder (named **Graphs**) for saving all generated visual outputs.


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

IMAGE_SAVE_DIR = "Graphs"
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)  # Create directory if it doesn't exist
```


**Why?**
- **Libraries:** They provide functions for everything from model loading to optimization.
- **Graph Directory:** Ensures that all plots are stored in one organized location.

---

### **Model and Scaler Loading**

The script dynamically finds the project root and loads the trained model (`best_model.h5`) and scaler (`scaler.pkl`). This makes the code more portable.


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
- **Dynamic Paths:** Using `Path` makes the code more robust across different systems.
- **Error Handling:** Ensures the script stops if essential files are missing.

---

## 3. Input Parsing and Prediction Functions

### **Parsing Forward Input**

The `parse_forward_input` function validates a user-provided comma-separated string. It expects exactly nine values (eight design parameters plus one frequency) and uses the `parse_frequency` helper to standardize the frequency value.


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
- **Validation:** Ensures that the correct number of inputs is provided.
- **Standardization:** Guarantees that the frequency is always in MHz, regardless of input format.

---

### **Frequency Parsing**

The `parse_frequency` function converts a frequency string (e.g., "2.4 GHz" or "2400 MHz") to a numeric value in MHz.


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
- **Regex Matching:** Captures both the numeric and unit parts.
- **Unit Conversion:** Converts GHz to MHz for uniformity.

---

### **Forward Prediction**

The `forward_predict` function scales the input parameters using the loaded scaler, feeds them to the model, and returns the prediction.


```
python
def forward_predict(input_params):
    """Scale input and predict output using the model."""
    if len(input_params) != 9:
        raise ValueError(f"Expected 9 features, got {len(input_params)}")
    input_arr = np.array(input_params, dtype='float32').reshape(1, -1)
    input_arr_scaled = scaler.transform(input_arr)
    pred = model(input_arr_scaled, training=False)
    return pred.numpy().flatten()
```


**Why?**
- **Scaling:** Ensures the input has the same distribution as the training data.
- **Model Inference:** Calls the model in inference mode (without training-specific behaviors like dropout).

---

### **Accuracy Calculation**

The `calculate_accuracy` function computes a normalized error between actual and predicted outputs, then converts it to an accuracy percentage.


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
- **Normalization:** Avoids division by zero and scales errors.
- **Interpretability:** Transforms error into a percentage that’s easier to understand.

---

## 4. Optimization Functions

### **Inverse Prediction**

The `inverse_predict` function optimizes the input parameters so that the model's output comes as close as possible to a desired target.


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
- **Objective Function:** Minimizes the squared difference between prediction and target.
- **Bounds & Clipping:** Ensures the optimization stays within physically or practically meaningful limits.
- **Optimization Method:** Uses L-BFGS-B for efficient, gradient-based local search.

---

## Extra Information: Inverse Mode Methods

**Definition & Purpose:**
Inverse prediction mode is used when you have a desired target output (for instance, specific S-parameters such as dB(S(1,1)) and dB(S(2,1))) and you want to determine the input design parameters that will produce outputs as close as possible to these targets. Essentially, it “inverts” the forward model.

**How It Works:**
- **Objective Function:**
  An objective function is defined to quantify the error between the neural network's predicted output and the desired target output. The function computes the sum of squared differences between each predicted value and its corresponding target.

- **Optimization with L-BFGS-B:**
  The inverse mode uses the L-BFGS-B optimization algorithm to minimize the objective function. L-BFGS-B (Limited-memory Broyden–Fletcher–Goldfarb–Shanno with Bound constraints) is a quasi-Newton method that:
  - Approximates the Hessian matrix (i.e., second derivatives) using limited memory.
  - Handles large-scale optimization problems efficiently.
  - Enforces bound constraints on the input parameters, ensuring that the optimized values remain within practical or physically meaningful ranges.

**Why It’s Used:**
- **Solving Nonlinear Inversion:**
  In many real-world applications, especially in antenna design, the relationship from input parameters to output performance is complex and nonlinear. Inverse prediction mode provides a way to back-calculate design parameters when the forward mapping (model) is known.
- **Bound Constraints:**
  By using L-BFGS-B, the method ensures that the optimized parameters do not exceed pre-defined limits, which is crucial in physical design scenarios where parameters must lie within specific ranges.

---

### **Dual-Frequency Optimization**

This section contains functions for optimizing design parameters over two frequencies simultaneously.

#### **Dual Objective Function**

The `dual_objective` function calculates a combined error for two frequencies. The target S-parameters are hard-coded (e.g., S11 = -10 dB and S21 = -1 dB).


```
python
def dual_objective(x, freq1, freq2):
    """
    Objective function for dual-frequency optimization.
    We target S11 = -10 dB and S21 = -1 dB at both frequencies.
    """
    target = np.array([-10, -1, -10, -1], dtype=np.float32)
    input_f1 = np.append(x, freq1).astype('float32').reshape(1, -1)
    input_f2 = np.append(x, freq2).astype('float32').reshape(1, -1)
    pred_f1 = model(scaler.transform(input_f1), training=False).numpy().flatten()
    pred_f2 = model(scaler.transform(input_f2), training=False).numpy().flatten()
    combined_pred = np.concatenate([pred_f1, pred_f2])
    return np.sum((combined_pred - target) ** 2)
```


**Why?**
- **Dual Targets:** Optimizes the design so that performance at both frequencies meets the targets.
- **Concatenation:** Combines predictions from both frequencies to compute a single error value.

#### **Global and Local Optimization Functions**

- **Global Optimization:** Uses `differential_evolution` to search broadly across the design space.

  
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


- **Local Optimization:** Uses L-BFGS-B to fine-tune the candidate solution found by the global optimizer.

  
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
- **Global vs. Local:** A global search helps escape local minima, while local refinement improves precision.
- **Optimization Strategies:** Differential evolution is robust for wide search spaces; L-BFGS-B is efficient for fine-tuning.

---

## Extra Information: Dual Mode Methods


### Dual-Frequency Optimization Mode

**Definition & Purpose:**
Dual-frequency optimization mode is designed to optimize antenna design parameters so that the antenna performs well at two distinct operating frequencies simultaneously. This is particularly important when an antenna must meet performance targets (e.g., S11 and S21 values) at multiple frequency bands.

**How It Works:**
- **Dual Objective Function:**
  A specialized objective function is defined to evaluate the performance of a candidate design across two frequencies. It:
  - Appends each frequency to the base design parameters.
  - Obtains predictions (S-parameters) from the neural network for each frequency.
  - Combines these predictions into a single error metric by calculating the sum of squared differences between the predicted and the target S-parameters for both frequencies.

- **Two-Stage Optimization Process:**
  1. **Global Optimization with Differential Evolution:**
     - **Differential Evolution (DE):**
       DE is a stochastic, population-based optimization algorithm well-suited for non-convex and high-dimensional problems. It broadly explores the design space to find promising candidates, reducing the risk of getting trapped in local minima.
  2. **Local Optimization with L-BFGS-B:**
     - Once a promising candidate is found using DE, L-BFGS-B is employed to fine-tune the design parameters, providing a more precise and optimized solution.

**Why It’s Used:**
- **Multiple Frequency Targets:**
  Antenna designs often need to achieve specific performance metrics across different frequency bands. Dual-frequency optimization ensures that the design meets targets at both frequencies simultaneously.
- **Combining Global and Local Methods:**
  - **Global Search (DE):** Helps explore a wide range of potential solutions in a complex, multi-modal landscape.
  - **Local Refinement (L-BFGS-B):** Fine-tunes the global candidate to achieve the best possible performance, ensuring precision.
- **Robust Performance:**
  This two-tiered approach increases the likelihood of finding a robust design that performs well under multiple operating conditions.

---

## 5. Visualization Functions

### **Design Parameter Formatting**

`format_design_parameters` produces a human-friendly string listing the design parameters.


```
python
def format_design_parameters(params):
    """Return a friendly formatted string of design parameters."""
    names = ['l_s', 'l_2', 'l_1', 's_2', 's_1', 'w_s', 'w_2', 'w_1']
    return ", ".join([f"{name}: {float(val):.3f}" for name, val in zip(names, params)])
```


**Why?**
- **Readability:** Helps users quickly see and compare design values in a standardized format.

### **Plotting Performance and Design Comparisons**

Two plotting functions create visual comparisons between initial and optimized designs:

- **plot_performance:** Shows S-parameter predictions across frequencies and compares initial versus optimized values.
- **plot_design_parameters:** Plots constant design parameters as horizontal lines versus frequency for visual consistency.

Both functions save their plots to the **Graphs** folder.


```
python
def plot_performance(initial_params, optimized_params, freq1, freq2):
    """Generate plots for frequency response vs frequency and S-parameter comparisons."""
    # (Plot code here, creating subplots for parameter bar charts and frequency response)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_SAVE_DIR,'dual_frequency_analysis.png'))
    plt.close()

def plot_design_parameters(initial_params, optimized_params):
    """Plot design parameters vs. frequency (displayed as constant horizontal lines)."""
    # (Plot code for lengths, spacings, and widths)
    plt.savefig(os.path.join(IMAGE_SAVE_DIR,'frequency_vs_width.png'))
    plt.close()
```


**Why?**
- **Visualization:** Plots provide intuitive comparisons and help validate the effectiveness of optimization.
- **File Organization:** Saving all plots in one folder keeps outputs organized.

---

## 6. Main Function and User Interaction

The `main()` function provides a command-line interface with three modes:
- **Dual-Frequency Optimization**
- **Forward Prediction**
- **Inverse Prediction**

Based on the user’s input, the script executes the corresponding branch.


```
python
def main():
    print("Welcome to the Enhanced Prediction System!")
    print("Choose a mode:")
    print("1. Forward Prediction")
    print("2. Inverse Prediction")
    print("3. Dual-Frequency Optimization")
    mode = input("Enter mode (forward/inverse/dual): ").strip().lower()

    if mode == "dual":
        # Dual-frequency mode: accepts 10 comma-separated values (8 design parameters + 2 frequencies)
        # Performs global then local optimization and displays predictions and error metrics.
        # Also generates performance and design parameter plots.
        # (Dual mode code here)

    elif mode == "forward":
        # Forward mode: accepts 9 comma-separated values and returns model prediction.
        input_str = input("Input: ")
        try:
            input_params = parse_forward_input(input_str)
            predicted = forward_predict(input_params)
            print(f"Model predicted output: {predicted}")
        except ValueError as e:
            print(f"Error: {e}")

    elif mode == "inverse":
        # Inverse mode: accepts target S-parameter values and an optional initial guess,
        # then optimizes input parameters to match the target.
        # (Inverse mode code here)

    else:
        print("Invalid mode. Please restart and choose 'forward', 'inverse', or 'dual'.")
```


**Why?**
- **User Interface:** Command-line prompts let users choose how to interact with the system.
- **Flexibility:** Supports different use cases (prediction vs. optimization) within one tool.

---

## 7. Final Thoughts

This advanced script demonstrates a complete pipeline for:
- **Loading and pre-processing design data,**
- **Predicting antenna S-parameters using a pre-trained model,**
- **Optimizing design parameters using both global and local search methods,**
- **Visualizing the results for easy interpretation.**


---