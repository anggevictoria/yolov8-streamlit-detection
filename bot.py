import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import requests

from modal import model_configuration_modal  # Import the modal function

# Main page heading
st.title("Object Detection")

# Button to open the model configuration modal
if st.button("Open Model Configuration"):
    # When the button is clicked, call the modal function
    model_configuration_modal()

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
        AIMessage(content="Hello, I am a bot. How can I help you?"),
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

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)