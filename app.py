# Python In-built packages
from pathlib import Path
import PIL

#Streamlit Streaming using LM Studio as OpenAI Standin
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

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

# Setting chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

# Main page heading
st.title("Object Detection")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Confidence Slider
confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Load Detection Model Only
model_path = Path(settings.DETECTION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None

# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
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
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_container_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    st.write("No image is uploaded yet!")

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

else:
    st.error("Please select a valid source type!")



# Add a space to separate sections visually
st.sidebar.markdown("---")

# Function to handle user queries and chat response
def get_response(user_query, chat_history):

    template = """
    You are a helpful assistant. Answer the following questions considering the history of the conversation:

    Chat history: {chat_history}

    User question: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Using LM Studio Local Inference Server
    llm = ChatOpenAI(api_key="YOUR_API_KEY", base_url="http://localhost:1234/v1/completion")


    chain = prompt | llm | StrOutputParser()
    
    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
    })

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

# Display conversation history in the sidebar
st.sidebar.header("Chat with the Assistant")
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.sidebar.chat_message("AI"):
            st.sidebar.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.sidebar.chat_message("Human"):
            st.sidebar.write(message.content)

# Ensure the chat input widget is at the bottom
user_query = st.sidebar.chat_input("Type your message here...")

if user_query:
    # Set up the chat model
    chat = ChatOpenAI(model="gemma-2-2b-it", temperature=0.7)

    # Create a chat prompt and generate response
    chat_prompt = ChatPromptTemplate.from_messages([HumanMessage(content=user_query)])
    response = chat(chat_prompt)

    # Append to chat history and display
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response.content))

    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            st.text(f"You: {message.content}")
        else:
            st.text(f"Bot: {message.content}")
