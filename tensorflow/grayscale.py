import tensorflow as tf
import numpy as np
import time
from timeit import timeit
from PIL import Image

def rgb2gray(input_data):
    input_data = tf.image.rgb_to_grayscale(input_data)
    input_data = tf.squeeze(input_data, axis=-1)
    input_data = tf.cast(input_data, dtype=tf.uint8)
    gray = input_data.numpy()
    return gray

#On Actual Image
input_image = Image.open("s-l1600.jpg")
input_data = tf.convert_to_tensor(np.array(input_image))
output_tensor = rgb2gray(input_data)
output_image = Image.fromarray(output_tensor, "L")
output_image.save("gray_cpu.jpg")


#Different Sizes
n=512

for i in range(1, 4):

    random_data = tf.random.normal((n, n, 3))
    input_data = tf.convert_to_tensor(random_data)
    print("Size = ", n, "x", n)

    n = n*4
     
    with tf.device('/CPU:0'):
        cpu_time = timeit(lambda: rgb2gray(input_data))

    with tf.device('/GPU:0'):
        gpu_time = timeit(lambda: rgb2gray(input_data))


    print("CPU time: ", cpu_time, " seconds")
    print("GPU time: ", gpu_time," seconds")
    speedup = cpu_time / gpu_time
    print("Speedup: ", speedup)

