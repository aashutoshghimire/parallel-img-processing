import tensorflow as tf
import time
from timeit import timeit
import numpy as np


def histogram_equalization_channel(channel):
    hist = tf.histogram_fixed_width(channel, value_range=(0.0, 1.0), nbins=256)
    cdf = tf.cumsum(hist)
    cdf = cdf / tf.reduce_max(cdf)
    equalized_channel = tf.gather(cdf, tf.cast(channel * 255.0, tf.int32))
    return equalized_channel

def histogram_equalization(r,g,b):
        r_equalized = histogram_equalization_channel(r)
        g_equalized = histogram_equalization_channel(g)
        b_equalized = histogram_equalization_channel(b)
        image_equalized = tf.concat([r_equalized, g_equalized, b_equalized], axis=-1)
        image_equalized = tf.cast(image_equalized * 255.0, tf.uint8)
        image_equalized = tf.clip_by_value(image_equalized, 0, 255)
        return image_equalized

#Histogram Equalization on Actual Image
image_path = 'image.jpg'
image = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image, channels=3)
image = tf.image.convert_image_dtype(image, tf.float32)
r, g, b = tf.split(image, 3, axis=-1)
#calling histogram equalization
image_equalized = histogram_equalization(r,g,b)
tf.io.write_file('histEq.jpg', tf.image.encode_jpeg(image_equalized))



#Running for different sizes
n=512

for i in range(1, 4):

     random_data = tf.random.uniform(shape=(n,n), minval=0.0, maxval=1.0)
     image = tf.image.convert_image_dtype(random_data, tf.float32)
     split_dim_size = random_data.shape[-1]
     split_size = split_dim_size // 3
     remainder = split_dim_size % 3
     split_sizes = [split_size] * 3
     split_sizes[0] += remainder
     r, g, b = tf.split(random_data, num_or_size_splits=split_sizes, axis=-1)
     print("Size = ", n, "x", n)

     n = n*4

     with tf.device('/CPU:0'):
        cpu_time = timeit(lambda: histogram_equalization(r,g,b))

     with tf.device('/GPU:0'):
        gpu_time = timeit(lambda: histogram_equalization(r,g,b))

     print("CPU time: ", cpu_time, " seconds")
     print("GPU time: ", gpu_time," seconds")
     speedup = cpu_time / gpu_time
     print("Speedup: ", speedup)
