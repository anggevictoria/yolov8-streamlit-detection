# Python In-built packages
from pathlib import Path
import PIL

#Streamlit Streaming using LM Studio as OpenAI Standing
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import requests

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="images/bot.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("VisionBot")

# Model Configuration in a Modal-like Component
with st.expander("⚙️ ML Model Configuration", expanded=False):
    st.subheader("Configure the Model")
    confidence = float(st.slider(
        "Select Model Confidence", 25, 100, 40)) / 100

    st.write(f"**Selected Confidence**: {confidence * 100}%")

    # Load Detection Model Only
    model_path = Path(settings.DETECTION_MODEL)

    # Load Pre-trained ML Model
    try:
        model = helper.load_model(model_path)
        st.success("Model successfully loaded!")
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)


# Image/Video Configuration
st.header("Image/Video Config")
source_radio = st.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None

# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_container_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_container_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_container_width=True)
        else:
            if st.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_container_width=True)
                
                detected_objects = set()  # Initialize an empty set to track detected object names

                try:
                    for box in boxes:
                        # Extract relevant information from the detection box
                        x1, y1, x2, y2, confidence, class_id = box.data[0][:6]  # Adjust indices if `box.data` has a different structure
                        object_name = model.names[int(class_id)]  # Map class ID to name

                        if object_name in detected_objects:
                            continue

                        #add the object to the set and process
                        detected_objects.add(object_name)            
                        description = helper.generate_description(object_name)
                        st.write(f"{object_name} detected: {description}")
                
                        
                except Exception as ex:
                    st.write("No image is uploaded yet!")

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

else:
    st.error("Please select a valid source.")


# Sidebar
st.sidebar.header("LM Studio Streaming Chatbot")

# Function to generate a description
def generate_description(object_name: str) -> str:
    api_url = "http://127.0.0.1:1234/v1/completions"
    payload = {
        "model": "gemma-2-2b-it",
        "prompt": f"Provide a brief description of a {object_name} in one sentence.",
        "temperature": 0.7,
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

def get_response(user_query, chat_history):
    # Modify the response to include a description generation example
    if "describe" in user_query.lower():
        object_name = user_query.split("describe")[-1].strip()
        description = generate_description(object_name)
        return f"Here is a brief description of {object_name}: {description}"

    template = """
    You are a helpful assistant. Use the chat history if it helps, otherwise ignore it:

    Chat history: {chat_history}

    User response: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Using LM Studio Local Inference Server
    llm = ChatOpenAI(openai_base_url="http://192.168.55.101:1234/v1/models")

    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
    })

# user input (placed before session state)
user_query = st.sidebar.text_input("Type your message here...")

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am VisionBot. How can I help you?"),
    ]

# Sidebar for displaying conversation history
with st.sidebar:
    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            response = get_response(user_query, st.session_state.chat_history)
            st.write(response)  # Use st.write instead of st.write_stream

        st.session_state.chat_history.append(AIMessage(content=response))