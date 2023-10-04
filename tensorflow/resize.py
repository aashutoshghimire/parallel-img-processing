import tensorflow as tf
import numpy as np
import time
from timeit import timeit

# Define the matrix dimensions
image_size = 8192

# Create a random image
image = tf.random.normal(shape=(1, image_size, image_size, 3), dtype=tf.float32)

# Define the TensorFlow function for bicubic interpolation
@tf.function
def resize_bicubic():
    # Apply the bicubic interpolation using tf.image.resize with cubic interpolation method
    resized_image = tf.image.resize(image, size=(256, 256), method=tf.image.ResizeMethod.BICUBIC)
    return resized_image

# Define the TensorFlow function for bilinear interpolation
@tf.function
def resize_bilinear():
    # Apply the bilinear interpolation using tf.image.resize with bilinear interpolation method
    resized_image = tf.image.resize(image, size=(256, 256), method=tf.image.ResizeMethod.BILINEAR)
    return resized_image

# Run the bicubic interpolation on the CPU and measure the time taken
with tf.device('/device:CPU:0'):
    cpu_time_bicubic = timeit(resize_bicubic)
    print("CPU time of Bicubic:", cpu_time_bicubic)

# Run the bicubic interpolation on the GPU and measure the time taken
with tf.device('/device:GPU:0'):
    gpu_time_bicubic = timeit(resize_bicubic)
    print("GPU time of Bicubic:", gpu_time_bicubic)

# Compute the speedup achieved by running the bicubic interpolation on the GPU
speedup_bicubic = cpu_time_bicubic / gpu_time_bicubic
print("Speedup time of Bicubic:", speedup_bicubic)


# Run the bilinear interpolation on the CPU and measure the time taken
with tf.device('/device:CPU:0'):
    cpu_time_bilinear = timeit(resize_bilinear)
    print("CPU time of Bilinear:", cpu_time_bilinear)


# Run the bilinear interpolation on the GPU and measure the time taken
with tf.device('/device:GPU:0'):
    gpu_time_bilinear = timeit(resize_bilinear)
    print("GPU time of Bilinear:", gpu_time_bilinear)



# Compute the speedup achieved by running the bilinear interpolation on the GPU
speedup_bilinear = cpu_time_bilinear / gpu_time_bilinear
print("Speedup of Bilinear:", speedup_bilinear)

