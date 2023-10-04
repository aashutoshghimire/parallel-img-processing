import tensorflow as tf
import time
from timeit import timeit

# Generate a 512x512 matrix with random values
matrix = tf.random.normal(shape=[512, 512])
#matrix = tf.random.normal(shape=[2048, 2048])
#matrix = tf.random.normal(shape=[4096, 4096])
#matrix = tf.random.normal(shape=[8192, 8192])

# Define the blur kernel
kernel = tf.constant([
    [1/16, 1/8, 1/16],
    [1/8, 1/4, 1/8],
    [1/16, 1/8, 1/16]
])

# Define the deblurring operation using matrix multiplication
def deblur():
    return tf.nn.conv2d(tf.expand_dims(tf.expand_dims(matrix, 0), -1),
                        tf.expand_dims(tf.expand_dims(kernel, -1), -1),
                        strides=[1, 1, 1, 1],
                        padding='SAME')[0, :, :, 0]

# Run the deblurring operation on CPU
with tf.device('/cpu:0'):
    cpu_time = timeit(deblur)
    print("Deblurring on CPU took {} seconds.".format(cpu_time))

# Run the deblurring operation on GPU
with tf.device('/gpu:0'):
    gpu_time = timeit(deblur)
    print("Deblurring on GPU took {} seconds.".format(gpu_time))

# Verify that the results are the same
#assert tf.reduce_all(tf.equal(deblur_cpu, deblur_gpu))

# Calculate the speedup
speedup = cpu_time/gpu_time
print("Speedup: {}x".format(speedup))
