# modal.py
import streamlit as st
from streamlit_modal import Modal

# Python In-built packages
from pathlib import Path
import PIL

#Streamlit Streaming using LM Studio as OpenAI Standing
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

def model_configuration_modal():
    """
    This function defines the modal for model configuration, which includes a model confidence slider.
    The modal will open only when the button is pressed.
    """
    modal = Modal("model_config", "Model Configuration")

    # Modal content
    with modal.container():
        st.header("Configure your Model")

        # Slider for selecting model confidence
        confidence = float(st.slider(
            "Select Model Confidence", 25, 100, 40)) / 100
        st.write(f"Model Confidence: {confidence}")

        # Other configuration settings
        model_name = st.text_input("Model Name", "gemma-2-2b-it")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
        max_tokens = st.number_input("Max Tokens", 1, 500, 50)

        if st.button("Save Configuration"):
            st.write(f"Configuration Saved: Model: {model_name}, Confidence: {confidence}, Temperature: {temperature}, Max Tokens: {max_tokens}")

    # Render the modal
    modal.render()
