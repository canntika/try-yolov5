import streamlit as st

st.set_page_config(page_title="Home",
                   layout='wide',
                    page_icon='./images/home.png' )

st.title("YOLO V5 Crack Detection App")
st.caption('This Web Application Demonstrate Crack Detection')

#Content
st.markdown("""
            ### This app detecs object from images
            - Automatically detects objects from image, video, and realtime detection
            - [Click here for image](/YOLO_for_image/)
            - [Click here for video](/YOLO_for_video/)
            - [Click here realtime](/YOLO_real-time/)
            
            below give are the object the our model will detect
            1. Retak Diagonal
            2. Retak Horizontal
            3. Retak Vertikal
            4. Retak Delta
            
            """)