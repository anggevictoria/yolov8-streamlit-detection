# image_video_config_modal.py
import streamlit as st
from streamlit_modal import Modal
import PIL
import settings
from helper import play_webcam  # Assuming this is a helper module for webcam functionality

def image_video_configuration_modal(model, confidence):
    """
    This function defines the modal for image/video configuration.
    """
    modal = Modal("image_video_config", "Image/Video Configuration")

    # Modal content
    with modal.container():
        st.header("Image/Video Config")
        
        # Source selection
        source_radio = st.radio(
            "Select Source", settings.SOURCES_LIST
        )

        source_img = None

        # If image is selected
        if source_radio == settings.IMAGE:
            source_img = st.file_uploader(
                "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp')
            )

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
                        res = model.predict(uploaded_image, conf=confidence)
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
            play_webcam(confidence, model)
        else:
            st.error("Please select a valid source type!")

    # Render the modal
    modal.render()
