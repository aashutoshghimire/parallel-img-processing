import tensorflow as tf
import time
from timeit import timeit
import numpy as np

kernel = tf.constant(tf.ones(shape=[ 5, 5, 1, 1]), dtype=tf.float32)

def convolution(matrix):
    return tf.nn.conv2d(matrix, kernel, strides=[1, 1, 1, 1], padding="VALID")

n=512

for i in range(1, 4):

    print("Size: ",n,"X",n)
    input_data = tf.random.normal([1, n, n, 1])

    n=n*4

    with tf.device('/CPU:0'):
        cpu_time = timeit(lambda: convolution(input_data))

    with tf.device('/GPU:0'):
        gpu_time = timeit(lambda: convolution(input_data))

    print("CPU Time", cpu_time)
    print("GPU Time", gpu_time)
    print("Speed Up :", cpu_time/gpu_time)
