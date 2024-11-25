from ultralytics import YOLO
import streamlit as st
import cv2
import settings
import requests

def load_model(model_path):
    model = YOLO(model_path)
    return model


def _display_detected_frames(conf, model, st_frame, image):
    # Resize the image to a fixed width of 160 pixels while maintaining a 16:9 aspect ratio
    aspect_ratio = 9 / 16
    new_width = 160
    new_height = int(new_width * aspect_ratio)
    image = cv2.resize(image, (new_width, new_height))
    model.to('cuda')
    
    # Perform object detection
    res = model.predict(image, conf=conf)

    # Plot the results
    res_plotted = res[0].plot()

    # Display the image with detected objects
    st_frame.image(res_plotted, caption='Detected Video', channels="BGR", use_container_width=True)

    # Extract detected object names
    object_names = set()
    boxes = res[0].boxes  # Access detected boxes (if available)
    if boxes:
        for box in boxes:
            class_id = int(box.cls)  # Get the class ID of the object
            object_name = model.names[class_id]  # Map class ID to object name
            object_names.add(object_name)

            st.write(f"{object_names}")

    return object_names
    
    


def play_webcam(conf, model):
    source_webcam = settings.WEBCAM_PATH
    if st.button('Detect Objects'):
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            detected_objects = set()  # Initialize a set to store unique object names

            while vid_cap.isOpened():
                

                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, model, st_frame, image)
                else:
                    vid_cap.release()
                    break

# extract the name of detected object
def get_detected_object_names(boxes, class_names):
    detected_objects = set()
    for box in boxes:
        class_id = int(box.cls)
        
        # Map class ID to object name
        object_name = class_names[class_id]
        detected_objects.add(object_name)
    return detected_objects

# Function to generate a description
def generate_description(object_name: str) -> str:
    api_url = "http://127.0.0.1:1234/v1/completions"
    payload = {
        "model": "gemma-2-2b-it",
        "prompt": f"Provide a simple description of {object_name} to a child in one sentence.",
        "temperature": 0.6,
        "max_tokens": 50
    }
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        data = response.json()
        description = data.get('choices', [{}])[0].get('text', "No description available.")
        first_sentence = description.split('.')[0].strip()
        return first_sentence
    except requests.exceptions.RequestException as e:
        # Improved error handling
        error_message = f"Error: {e}"
        if 'response' in locals():
            error_message += f", Response: {response.text}"
        else:
            error_message += ", No response"
        return error_message
