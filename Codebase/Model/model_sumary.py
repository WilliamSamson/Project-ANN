import os
import json
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# Function to check for GPU availability
def check_device():
    physical_devices = tf.config.list_physical_devices('GPU')
    device_info = {"device_count": len(physical_devices), "devices": []}
    if physical_devices:
        for device in physical_devices:
            device_info["devices"].append(device.name)
    else:
        device_info["devices"].append("No GPU available, using CPU.")
    return device_info

# Define paths
model_path = os.path.expanduser("~/PycharmProjects/Project ANN/Codebase/Model/VER_3/v_1/best_model.h5")
output_dir = os.path.expanduser("~/PycharmProjects/Project ANN/Codebase/Model")
os.makedirs(output_dir, exist_ok=True)

# Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Custom objects mapping to handle 'mse'
custom_objects = {'mse': tf.keras.losses.MeanSquaredError()}

# Load the model with custom objects
model = load_model(model_path, custom_objects=custom_objects)
print("Model loaded successfully.")

# Save the basic model summary to a text file
model_summary_path = os.path.join(output_dir, "model_summary.txt")
with open(model_summary_path, "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))
print(f"Model summary saved to {model_summary_path}")

# Save a visualization of the model architecture
plot_path = os.path.join(output_dir, "model_architecture.png")
plot_model(model, to_file=plot_path, show_shapes=True, show_layer_names=True)
print(f"Model architecture visualization saved as {plot_path}")

# Optional: Evaluate the model on a test set
def evaluate_model(model, test_data, test_labels, output_file):
    results = model.evaluate(test_data, test_labels, verbose=1)
    metrics = {metric: value for metric, value in zip(model.metrics_names, results)}
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Model evaluation results saved to {output_file}")
    return metrics

# Optional: Make predictions
def make_predictions(model, data, output_file, label_encoder=None):
    predictions = model.predict(data)
    if label_encoder:
        predictions = label_encoder.inverse_transform(predictions.argmax(axis=1))
    else:
        predictions = predictions.tolist()
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=4)
    print(f"Predictions saved to {output_file}")
    return predictions

# Save GPU/CPU information to a file
device_info = check_device()
device_info_path = os.path.join(output_dir, "device_info.json")
with open(device_info_path, "w") as f:
    json.dump(device_info, f, indent=4)
print(f"Device info saved to {device_info_path}")

# Generate a comprehensive HTML report of the model
def generate_html_report(model, report_path, plot_image_path, device_info):
    # Capture the model summary as a string
    summary_str_io = io.StringIO()
    model.summary(print_fn=lambda x: summary_str_io.write(x + "\n"))
    summary_text = summary_str_io.getvalue()

    # Retrieve optimizer details
    if hasattr(model, 'optimizer'):
        optimizer = model.optimizer
        optimizer_name = type(optimizer).__name__
        try:
            lr = optimizer.learning_rate.numpy() if hasattr(optimizer.learning_rate, 'numpy') else optimizer.learning_rate
        except Exception:
            lr = optimizer.learning_rate
        optimizer_details_html = f"<p><strong>Optimizer:</strong> {optimizer_name}</p><p><strong>Learning Rate:</strong> {lr}</p>"
    else:
        optimizer_details_html = "<p>No optimizer information available.</p>"

    # Build per-layer details (name, type, activation function, output shape, and parameters)
    layer_rows = ""
    for layer in model.layers:
        config = layer.get_config()
        activation = config.get("activation", "N/A")
        layer_type = layer.__class__.__name__
        try:
            output_shape = layer.output_shape
        except Exception:
            output_shape = "N/A"
        param_count = layer.count_params()
        layer_rows += f"<tr><td>{layer.name}</td><td>{layer_type}</td><td>{activation}</td><td>{output_shape}</td><td>{param_count}</td></tr>"

    layer_details_html = f"""
    <table>
        <tr>
            <th>Layer Name</th>
            <th>Layer Type</th>
            <th>Activation</th>
            <th>Output Shape</th>
            <th>Parameters</th>
        </tr>
        {layer_rows}
    </table>
    """

    # Get the model configuration as a JSON string (if possible)
    try:
        model_config = json.dumps(model.get_config(), indent=4)
    except Exception:
        model_config = "Model configuration could not be serialized."

    # Note on epochs (not stored in the model file)
    epochs_info = "Number of epochs information not available in the saved model file."

    # Construct the full HTML content
    html_content = f"""
    <html>
    <head>
        <title>Comprehensive Model Summary Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2 {{ color: #333; }}
            pre {{ background-color: #f4f4f4; padding: 10px; border: 1px solid #ddd; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .section {{ margin-bottom: 40px; }}
        </style>
    </head>
    <body>
        <h1>Comprehensive Model Summary Report</h1>
        <div class="section">
            <h2>Model Summary</h2>
            <pre>{summary_text}</pre>
        </div>
        <div class="section">
            <h2>Optimizer Details</h2>
            {optimizer_details_html}
        </div>
        <div class="section">
            <h2>Layer Details</h2>
            {layer_details_html}
        </div>
        <div class="section">
            <h2>Model Configuration</h2>
            <pre>{model_config}</pre>
        </div>
        <div class="section">
            <h2>Epochs Information</h2>
            <p>{epochs_info}</p>
        </div>
        <div class="section">
            <h2>Device Information</h2>
            <pre>{json.dumps(device_info, indent=4)}</pre>
        </div>
        <div class="section">
            <h2>Model Architecture Visualization</h2>
            <img src="{plot_image_path}" alt="Model Architecture Diagram" style="max-width:100%;">
        </div>
    </body>
    </html>
    """

    # Write the HTML report to file
    with open(report_path, "w") as f:
        f.write(html_content)
    print(f"Comprehensive model summary report saved to {report_path}")

# Generate the HTML report
html_report_path = os.path.join(output_dir, "comprehensive_model_summary.html")
generate_html_report(model, html_report_path, plot_path, device_info)
