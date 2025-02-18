import os
import json
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

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


project_root = Path(__file__).resolve().parents[1]  # Go up 4 levels from script directory
data_path = project_root / "best_model.h5"

if not data_path.exists():
   raise FileNotFoundError(f"File not found: {data_path}")

# Custom objects mapping to handle 'mse'
custom_objects = {'mse': tf.keras.losses.MeanSquaredError()}

# Load the model with custom objects
model = load_model(str(data_path), custom_objects=custom_objects)
print("Model loaded successfully.")

# Save the basic model summary to a text file

script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script's directory
model_summary_path = os.path.join(script_dir, "model_summary.txt")
with open(model_summary_path, "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))
print(f"Model summary saved to {model_summary_path}")


# -----------------------
# Native visualization of model architecture using matplotlib
def draw_model_architecture(model, save_path):
    """
    Creates a simple vertical diagram of the model architecture.
    NOTE: This function currently supports sequential models.
    """
    layers = model.layers
    n_layers = len(layers)

    # Parameters for drawing
    box_width = 6
    box_height = 1
    margin = 0.5
    x0 = 1
    # Calculate starting y coordinate so that the diagram fits well
    y0 = (n_layers) * (box_height + margin) + margin

    # Create a new figure
    fig, ax = plt.subplots(figsize=(8, y0 / 1.5))
    ax.axis('off')

    positions = []
    for i, layer in enumerate(layers):
        # Compute the lower left corner (x, y) for each box
        y = y0 - (i + 1) * (box_height + margin)
        positions.append(y)

        # Extract layer details
        config = layer.get_config()
        activation = config.get("activation", "N/A")
        layer_info = f"{layer.name}\nType: {layer.__class__.__name__}\nActivation: {activation}"

        # Draw the rectangle for the layer
        rect = patches.Rectangle((x0, y), box_width, box_height, linewidth=1, edgecolor='black', facecolor='skyblue')
        ax.add_patch(rect)
        ax.text(x0 + box_width / 2, y + box_height / 2, layer_info, ha='center', va='center', fontsize=10)

        # Draw an arrow connecting to the previous layer (if not the first)
        if i > 0:
            prev_y = positions[i - 1]
            start_x = x0 + box_width / 2
            start_y = prev_y  # bottom center of previous box
            end_y = y + box_height  # top center of current box
            dy = end_y - start_y
            ax.arrow(start_x, start_y, 0, dy, head_width=0.2, head_length=0.2, fc='k', ec='k',
                     length_includes_head=True)

    # Adjust limits and save the figure
    ax.set_xlim(0, x0 + box_width + 1)
    ax.set_ylim(0, y0 + margin)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Native model architecture visualization saved as {save_path}")

output_dir = os.path.dirname(os.path.abspath(__file__))  # Get script's directory
# Generate and save the native visualization of the model architecture
plot_path = os.path.join(output_dir, "model_architecture_native.png")
draw_model_architecture(model, plot_path)


# Optional: Evaluate the model on a prediction_script set
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

    # Retrieve optimizer details including learning rate
    if hasattr(model, 'optimizer'):
        optimizer = model.optimizer
        optimizer_name = type(optimizer).__name__
        try:
            lr = optimizer.learning_rate.numpy() if hasattr(optimizer.learning_rate,
                                                            'numpy') else optimizer.learning_rate
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
