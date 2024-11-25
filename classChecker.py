# Python In-built packages
from pathlib import Path

# Local Modules
import settings
import helper

model_path = Path(settings.DETECTION_MODEL)

# Load Pre-trained YOLO Model
model = helper.load_model(model_path)

# Check if model.names exists
if hasattr(model, 'names'):
    class_names = model.names
    print("Class names loaded successfully.")
else:
    print("Error: Model does not have 'names' attribute.")

# class names 
print("Class names:", class_names)