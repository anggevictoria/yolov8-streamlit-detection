from ultralytics import YOLO
import streamlit as st
import cv2
import settings
import requests
import time

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

    # Extract detected object names every 3 seconds
    object_names_set = set()
    boxes = res[0].boxes  # Access detected boxes (if available)
    
    current_time = time.time()  # Get current time
    if current_time - _display_detected_frames.last_execution_time > 5:  # Check if 3 seconds have passed
        _display_detected_frames.last_execution_time = current_time  # Update the last execution time

        if boxes:
            for box in boxes:
                class_id = int(box.cls)  # Get the class ID of the object
                object_name = model.names[class_id]  # Map class ID to object name
                if object_name not in object_names_set:  # Check if the name is already in the set
                    object_names_set.add(object_name)  # Add to the set
                st.write(f"Detected Objects (Set): {object_names_set}")
                
                object_names_list = list(object_names_set) #convert set to list for order
                
                object_names_set.clear()
                st.write(f"Detected Objects (Set): {object_names_set}")
                st.write(f"Detected Objects (List): {object_names_list}")
                
        st.write(f"Detected Objects(Set): {object_names_set}")  # Display the detected objects in Streamlit

    return object_names_set

# Initialize the last execution time attribute for the function
_display_detected_frames.last_execution_time = 0  # First execution is allowed immediately


def play_webcam(conf, model):
    source_webcam = settings.WEBCAM_PATH
    if st.button('Detect Objects'):
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()

            while vid_cap.isOpened():
                

                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, model, st_frame, image)
                else:
                    vid_cap.release()
                    break

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
