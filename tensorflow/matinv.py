import tensorflow as tf
from timeit import timeit

def matrixInverse(matri):
    return tf.linalg.inv(matrix)

n=512

for i in range(1, 4):

    matrix = tf.random.normal((n,n))
    print("Size: ", n, 'X', n)

    n = n*4
    with tf.device('/CPU:0'):
        cpu_time = timeit(lambda: matrixInverse(matrix))

    with tf.device('/GPU:0'):
        gpu_time = timeit(lambda: matrixInverse(matrix))


    print("CPU time: ", cpu_time, " seconds")
    print("GPU time: ", gpu_time," seconds")
    speedup = cpu_time / gpu_time
    print("Speedup: ", speedup)
