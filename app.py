import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load the trained CNN model
model = load_model("mnist_cnn_model.h5")

st.title("Handwritten Digit Recognizer")
st.write("Draw a digit below and let the model predict it!")

# Create a canvas for drawing
canvas_result = st_canvas(
    fill_color="black",  # Background color
    stroke_width=10,
    stroke_color="white",  # Digit color
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

def preprocess_image(image):
    image = cv2.resize(image, (28, 28))  # Resize to match MNIST dataset
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.bitwise_not(image)  # Invert colors (MNIST uses white digits on black background)
    image = image / 255.0  # Normalize
    image = image.reshape(1, 28, 28, 1)  # Reshape for model input
    return image

if canvas_result.image_data is not None:
    img = canvas_result.image_data.astype(np.uint8)
    processed_image = preprocess_image(img)
    
    # Make prediction
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)
    st.write(f"### Predicted Digit: {predicted_digit}")