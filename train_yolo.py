import os
import matplotlib.pyplot as plt

from ultralytics import YOLO

# Dataset paths
dataset_path = "utility\dataset_yolo"
data_yaml = os.path.join(dataset_path, "data.yaml")

# Load pretrained YOLO model
model_name = "yolov8n.pt"
model = YOLO(model_name)  # Load a pretrained YOLO model.

# Train the model with frozen layers
results = model.train(
    data=data_yaml,       # Dataset configuration
    epochs=50,            # Number of epochs
    batch=16,             # Batch size
    imgsz=640,            # Image size
    device=0,             # Use GPU (0 for first GPU)
    freeze=[0, 10]        # Freeze first 10 layers (adjust as needed)
)

# Extract training history
train_loss = results.metrics['train/loss']  # Training loss
val_loss = results.metrics['val/loss']      # Validation loss

# Save the trained model
output_model_path = f"{model_name}" + ".pt"
model.save(output_model_path)
print(f"Model saved as {output_model_path}")

# Plot the loss
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Training Loss', marker='o')
plt.plot(val_loss, label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Save the plot
plt.savefig(f'{model_name}_training_vs_validation_loss.png')
