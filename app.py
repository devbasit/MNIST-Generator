import streamlit as st
import tensorflow as tf
import numpy as np
from keras import layers, models
import matplotlib.pyplot as plt

# Load the TFLite model
try:
    interpreter = tf.lite.Interpreter(model_path="generator.tflite")
    interpreter.allocate_tensors()
except:
    st.error("TFLite model 'generator.tflite' not found. Please run the previous cells to generate and save the model.")
    st.stop() # Stop the script if the model is not found

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to generate digit image from integer input using TFLite model
def generate_digit_tflite(digit):
    if not 0 <= digit <= 9:
        raise ValueError("Digit must be between 0 and 9")

    # Create one-hot encoded input
    input_vector = (tf.keras.utils.to_categorical([digit], 10)+np.random.normal(0, 0.1, (1, 10))).astype(np.float32)
    input_data = np.array(input_vector, dtype=np.float32)

    # Set the tensor to the input data
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    generated_image = output_data[0]

    return generated_image.reshape(28, 28)

# Streamlit App
st.title("Handwritten MNIST Image Generator")

# Dropdown for digit selection
selected_digit = st.selectbox("Choose a digit:", options=list(range(10)))

# Submit button
if st.button("Generate Images"):
    st.write(f"Generating images for digit {selected_digit}...")

    # Generate five images
    generated_images = [generate_digit_tflite(selected_digit) for _ in range(5)]

    # Display images side by side
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image(generated_images[0], clamp=True, output_format="png")
    with col2:
        st.image(generated_images[1],  clamp=True, output_format="png")
    with col3:
        st.image(generated_images[2],  clamp=True, output_format="png")
    with col4:
        st.image(generated_images[3],  clamp=True, output_format="png")
    with col5:
        st.image(generated_images[4],  clamp=True, output_format="png")

    st.success("Images generated!")
