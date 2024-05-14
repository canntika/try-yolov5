import streamlit as st
from yolo_predictions import YOLO_Pred
from PIL import Image
import numpy as np

st.set_page_config(page_title="YOLO Object Detection",
                   layout='wide',
                    page_icon='./images/object.png' )

st.title('WELCOME TO CRACK DETECTION FOR IMAGE')
st.write('Please Upload Image to Get Detections')

with st.spinner('Please wait while your model is loading'):
    yolo = YOLO_Pred(onnx_model='./models/best.onnx',
                    data_yaml='./models/data.yaml')
    ##st.balloons()


def upload_image():
    #Upload Image
    image_file = st.file_uploader(label='Upload Image')
    if image_file is not None:
        size_mb = image_file.size/(1024**2)
        file_details ={"filename":image_file.name,
                    "filetype":image_file.type,
                    "filesize":"{:,.2f} MB".format(size_mb)}
        ##st.json(file_details)
        #validate file
        if file_details['filetype'] in ('image/png','image/jpeg'):
            st.success('VALID IMAGE file type (png or jpeg)')
            return {"file":image_file, 
                    "details":file_details}
        else:
            st.error('INVALID Image file type')
            st.error('Upload only png,jpg,jpeg')
            return None
        
def main():
    object = upload_image()

    if object:
        prediction = False
        image_obj = Image.open(object['file']).convert('RGB')
        

        col1 , col2 = st.columns(2)

        with col1:
            st.info('Preview of Image')
            st.image(image_obj)

        with col2:
            st.subheader('Chech below for file details')
            st.json(object['details'])
            button = st.button('Get Detection from YOLO')
            if button:
                with st.spinner("""
                Getting Objects from image. please wait
                                """):
                #below command will convert obj to array
                    image_array = np.array(image_obj)
                    pred_img= yolo.predictions(image_array)
                    pred_img_obj = Image.fromarray(pred_img)

                     # Perform object detection using YOLO
                    #detections = yolo.predictions(image_obj)
                    #print(detection)
                    #detected_objects = {}
                    #for detection in detections:
                    #    object_class = detection['classes']
                    #    detected_objects[object_class] = detected_objects.get(object_class, 0) + 1

                    prediction = True

        if prediction:
            st.subheader("Prediction Image")
            st.caption("Object Detections Complete")
            st.image(pred_img_obj)
            st.write('Real-time Object Detection Results:')
           # st.write('Retak Diagonal =' +str(my_array[0]))
          #  st.write('Retak Vertikal =' +str(my_array[1]))
          #  st.write('Retak Horizontal =' +str(my_array[2])) 
            #for object_class, count in detected_objects.items():
            #    st.write(f'{object_class}: {count}')
           
        
if __name__ == "__main__":
    main()