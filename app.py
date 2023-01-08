import streamlit as st
import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image
import numpy as np
from image_classifier import image_label

st.title("Image Classifier \n (upload any image of building, forest, glacier, mountain, sea, and street)")


if __name__ == "__main__":
    uploaded_file = st.file_uploader("Choose a image file")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        image = np.array(image)
        img_input = np.array(tf.image.resize(image, (150, 150), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)).reshape((1, 150, 150, 3))
        st.header(image_label(img_input))
	    # st.success(image_label(img_input))
