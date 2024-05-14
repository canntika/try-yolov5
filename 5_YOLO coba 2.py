import streamlit as st
from yolo_predictions import YOLO_Pred
import cv2
import tempfile
import numpy as np

st.set_page_config(page_title="YOLO Object Detection", layout='wide', page_icon='./images/object.png')

st.title('WELCOME TO CRACK DETECTION FOR VIDEO')
st.write('Please Upload Video to Get Detections')

# Initialize YOLO model
with st.spinner('Please wait while crack detction model is loading'):
    yolo = YOLO_Pred(onnx_model='./models/best.onnx', data_yaml='./models/data.yaml')

def upload_video():
    video_file = st.file_uploader(label='Upload Video', type=['mp4', 'avi'])
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(video_file.read())
        return tfile.name
    return None

def process_video(video_path):
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    # Placeholder for displaying frames
    frame_placeholder = st.empty()
    # Dictionary to accumulate object counts
    object_counts = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Process frame
        pred_frame, my_array = yolo.predictions(frame)
        # Update object counts
        #for obj in detected_objects:
        #    if obj in object_counts:
        #        object_counts[obj] += 1
        #    else:
        #        object_counts[obj] = 1
        # Convert color from BGR to RGB
        frame_rgb = cv2.cvtColor(pred_frame, cv2.COLOR_BGR2RGB)
        # Display frame
        frame_placeholder.image(frame_rgb)
    cap.release()
    return my_array

def main():
    video_path = upload_video()

    if video_path:
        col1, col2 = st.columns(2)

        with col1:
            st.info('Preview of Video')
            st.video(video_path)

        with col2:
            st.subheader('Check below for file details')
            # Display video details if needed

            button = st.button('Get Detection from YOLO')
            if button:
                with st.spinner("Getting Objects from video. Please wait..."):
                    my_array = process_video(video_path)
                    st.success("Object Detections Complete")
                    # Display object counts
                    st.subheader("Detected Objects Count:")
                    #for obj, count in object_counts.items():
                    st.write('Retak Diagonal =' +str(my_array[0]))
                    st.write('Retak Vertikal =' +str(my_array[1]))
                    st.write('Retak Horizontal =' +str(my_array[2])) 

if __name__ == "__main__":
    main()
