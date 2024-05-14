import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
from yolo_predictions import YOLO_Pred

# Load YOLO model
yolo = YOLO_Pred('./models/best.onnx', './models/data.yaml')

# Define the keys for the object classes
OBJECT_KEYS = ['retak diagonal', 'retak vertikal', 'retak horizontal']

# Initialize object count dictionary in session state if it doesn't exist
if 'object_counts' not in st.session_state:
    st.session_state.object_counts = {key: 0 for key in OBJECT_KEYS}

class YOLOVideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Predict objects in the image
        pred_img, my_array = yolo.predictions(img)

        # Update object counts based on predictions
        st.session_state.object_counts['retak diagonal'] += my_array[1]
        st.session_state.object_counts['retak vertikal'] += my_array[2]
        st.session_state.object_counts['retak horizontal'] += my_array[3]

        # Convert the image with predictions to video frame
        new_frame = av.VideoFrame.from_ndarray(pred_img, format="bgr24")
        return new_frame

# Setup the layout with two columns
col1, col2 = st.columns(2)

with col1:
    st.header("Object Detection")
    # Start the camera stream using webrtc_streamer
    webrtc_ctx = webrtc_streamer(key="example",
                                 video_processor_factory=YOLOVideoProcessor,
                                 media_stream_constraints={"video": True, "audio": False})

with col2:
    st.header("Detected Objects Count")
    # Display the object counts
    if webrtc_ctx.video_receiver:
        for key, value in st.session_state.object_counts.items():
            st.write(f"{key}: {value}")
    else:
        st.write("Camera is not active or no objects detected.")
