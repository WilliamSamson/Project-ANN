import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import json

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
model_path = os.path.expanduser("~/PycharmProjects/Project ANN/Codebase/Model/VER_2/best_model_advanced.h5")
output_dir = os.path.expanduser("~/PycharmProjects/Project ANN/Codebase/Model/VER_2/")
os.makedirs(output_dir, exist_ok=True)

# Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Training_set file not found at {model_path}")

# Load the model
model = load_model(model_path)
print("Training_set loaded successfully.")

# Export the model summary to a file
model_summary_path = os.path.join(output_dir, "model_summary.txt")
with open(model_summary_path, "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))
print(f"Training_set summary saved to {model_summary_path}")

# Save a visualization of the model architecture
plot_path = os.path.join(output_dir, "model_architecture.png")
plot_model(model, to_file=plot_path, show_shapes=True, show_layer_names=True)
print(f"Training_set architecture saved as {plot_path}")

# Optional: Evaluate the model on a test set
def evaluate_model(model, test_data, test_labels, output_file):
    results = model.evaluate(test_data, test_labels, verbose=1)
    metrics = {metric: value for metric, value in zip(model.metrics_names, results)}
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Training_set evaluation results saved to {output_file}")
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
