import requests
import streamlit as st
from PIL import Image
from yolo_predictions import YOLO_Pred
import cv2
from io import BytesIO
import av
import numpy as np
from streamlit_webrtc import (
    ClientSettings,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
)

st.set_page_config(page_title="Computer vision", page_icon="🖥️")

# load yolo model
yolo = YOLO_Pred('./models/best.onnx',
                 './models/data.yaml')

def main():
    st.title("Object Detection App")

    # Create a radio button to choose between uploading an image, using the webcam or providing an image URL
    choice = st.radio("Select an option", ("Upload an image", "Use webcam", "Provide image URL"))

    if choice == "Upload an image":
        # Create a file uploader widget
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        # If a file is uploaded
        if uploaded_file is not None:
            # Load the image from the uploaded file
            img = Image.open(uploaded_file)

            #image = yolo.predictions(img)
            #res_plotted = image[0].plot()
            #cv2.imwrite('images/test_image_output.jpg', res_plotted)
            image_array = np.array(img)
            pred_img,my_array= yolo.predictions(image_array)
            pred_img_obj = Image.fromarray(pred_img)
            
            col1, col2 = st.columns(2)

            col1.image(img, caption="Uploaded Image", use_column_width=True)
            # Display the uploaded image
            col2.image(pred_img_obj, caption="Predected Image", use_column_width=True)

    elif choice == "Use webcam":
        # Define the WebRTC client settings
        client_settings = ClientSettings(
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={
                "video": True,
                "audio": False,
            },
        )

        # Define the WebRTC video transformer
        class ObjectDetector(VideoTransformerBase):
            def transform(self,frame):
                # Convert the frame to an image
                img = frame.to_ndarray(format="bgr24")

                image, my_array = yolo.predictions(img)
                #res_plotted = image[0].plot()
                output_frame = av.VideoFrame.from_ndarray(image, format="bgr24")
                #cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

                    # Return the annotated frame
                return output_frame

        # Start the WebRTC streamer
        webrtc_streamer(
            key="object-detection",
            client_settings=client_settings,
            video_transformer_factory=ObjectDetector
        )

    elif choice == "Provide image URL":
        # Get the image URL from the user
        image_url = st.text_input("Enter the image URL:")

        # If the user has entered an image URL
        if image_url != "":
            try:
                # Download the image from the URL
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content))

                #image = yolo.predictions(img)
                image_array = np.array(img)
                pred_img,my_array= yolo.predictions(image_array)
                pred_img_obj = Image.fromarray(pred_img)

                #res_plotted = image[0].plot()
                #cv2.imwrite('images/test_image_output.jpg', res_plotted)
                
                col1, col2 = st.columns(2)
                col1.image(img, caption="Downloaded Image" , use_column_width=True)
                # Display the downloaded image
                col2.image(pred_img_obj, caption="predected Image", use_column_width=True)
            except:
                st.error("Error: Invalid image URL or unable to download the image.")


if __name__ == '__main__':
    main()