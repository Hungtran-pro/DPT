import streamlit as st
from utils import prediction
import cv2
import numpy as np

def predict_function(img, method):
    return prediction(img, method)

# Set up the app layout
st.set_page_config(page_title="Upload File Demo")
st.title("Phân loại giới tính Demo")

# Set up the upload file section
uploaded_file = st.file_uploader("Please upload an image", type=["jpg", "jpeg", "png"])

# Set up the options selection section
option_selected = st.selectbox("Select a feature extraction method", ["Ratio hair", "HOG", "Both HOG and ratio"])

# Set up the Predict button
if st.button("Predict") and uploaded_file is not None:

    st.write("You uploaded the file:", uploaded_file.name)
    
    if option_selected == "Ratio hair":
        st.write("You selected 'Ratio of hair' method")
    elif option_selected == "HOG":
        st.write("You selected 'HOG' method.")
    elif option_selected == "Both HOG and ratio":
        st.write("You selected 'Both HOG and ratio' method.")

    # Call your prediction function here
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    prediction = predict_function(image, option_selected)
    st.write("The predicted gender is:", prediction)
    st.image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB), caption='Uploaded Image')