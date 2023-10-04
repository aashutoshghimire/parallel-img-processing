import tensorflow as tf
import numpy as np
import time
from timeit import timeit
from PIL import Image

# Define the matrix dimensions
image_size = 512

# Load and preprocess the input image
input_image = Image.open('grayscalecoin.jpg')  # Replace 'input.jpg' with the path to your input image
input_image = input_image.resize((image_size, image_size))
input_image = input_image.convert('L')  # Convert to grayscale
input_image = np.array(input_image)
input_image = input_image.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
input_image = np.expand_dims(input_image, axis=-1)  # Add channel dimension
image = tf.constant(input_image)

# Define the Sobel edge detection function
@tf.function
def sobel_edge_detection():
    # Apply the Sobel edge detection using tf.image.sobel_edges
    gradients = tf.image.sobel_edges(image)
    edges = tf.sqrt(gradients[..., 0] ** 2 + gradients[..., 1] ** 2)  # Compute edge map
    return edges

# Run the Sobel edge detection on the CPU and measure the time taken
with tf.device('/device:CPU:0'):
    cpu_time = timeit(sobel_edge_detection)
    print("CPU time:", cpu_time)

# Run the Sobel edge detection on the GPU and measure the time taken
with tf.device('/device:GPU:0'):
    gpu_time = timeit(sobel_edge_detection)
    print("GPU time:", gpu_time)

# Compute the speedup achieved by running the Sobel edge detection on the GPU
speedup = cpu_time / gpu_time
print("Speedup:", speedup)

# Get the output image as a NumPy array
output_image = sobel_edge_detection().numpy()[0, ..., 0]  # Remove the channel dimension

# Convert the output image to PIL Image format
output_image = (output_image * 255).astype(np.uint8)
output_image = Image.fromarray(output_image, mode='L')  # Specify mode='L' for grayscale image

# Save the output image
output_image.save('output.jpg')  # Replace 'output.jpg' with the desired output image file path
print("Output image saved as 'output.jpg'")

