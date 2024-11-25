import cv2
from ultralytics import YOLO
import time

# Load the YOLO model
model = YOLO("yolov8m.pt")  # Ensure the correct model filename

# Open the default webcam
cap = cv2.VideoCapture(0)

# Track detected objects from the previous frame
previously_detected_objects = set()

# Store the last time the print statement was called
last_print_time = time.time()

# Function to extract names of detected objects from boxes
def get_detected_object_names(boxes, class_names):
    detected_objects = set()
    for box in boxes:
        class_id = int(box.cls)
        # Map class ID to object name
        object_name = class_names[class_id]
        detected_objects.add(object_name)
    return detected_objects

# Loop through the video frames from the webcam
while cap.isOpened():
    # Read a frame from the webcam
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model.predict(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Extract class names and boxes from results
        boxes = results[0].boxes
        class_names = model.names

        # Get the detected object names
        current_frame_objects = get_detected_object_names(boxes, class_names)

        # Display the annotated frame
        cv2.imshow("YOLO Webcam Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the frame could not be read
        break

# Release the webcam and close the display window
cap.release()
cv2.destroyAllWindows()
