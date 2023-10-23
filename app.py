import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image
#load the CNN model
model = tf.keras.models.model_from_json(open('model.json', 'r').read())
model.load_weights('model_weights.h5')

st.title("Cat vs. Dog Classifier")

# Upload an image for classification
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert the image to a NumPy array and resize it
    image = np.array(image)
    image = tf.image.resize(image, (256,256))

    # Normalize the image pixel values
    image = image / 255.0

    # Expand the dimensions for model input
    image = np.expand_dims(image, axis=0)

    # Make predictions
    prediction = model.predict(image)

     # Interpret the prediction
    if prediction[0][0] >= 0.7:
        st.title("Prediction: Dog")
    elif prediction[0][0] <= 0.6:
        st.title("Prediction: Cat")
