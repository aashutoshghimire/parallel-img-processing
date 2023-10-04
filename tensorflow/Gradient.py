import tensorflow as tf
import numpy as np
import time

# generate random 3D vectors
num_vecs = 10000
vecs = tf.constant(np.random.normal(size=(num_vecs,3)))

# define the gradient function
@tf.function
def gradient(vecs):
    with tf.GradientTape() as tape:
        tape.watch(vecs)
        out = tf.reduce_sum(tf.square(vecs), axis=-1)
    return tape.gradient(out, vecs)

# CPU calculation
start_time = time.time()
cpu_grad = gradient(vecs)
cpu_time = time.time() - start_time
print("CPU time:", cpu_time, "s")

# GPU calculation
with tf.device('/GPU:0'):
    start_time = time.time()
    gpu_grad = gradient(vecs)
    gpu_time = time.time() - start_time
print("GPU time:", gpu_time, "s")

# calculate speedup
speedup = cpu_time / gpu_time
print("Speedup:", speedup)
