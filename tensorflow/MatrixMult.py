import tensorflow as tf
import sys
from timeit import timeit

def matrixMultiplication(matrix1, matrix2):
    return tf.matmul(matrix1, matrix2)

n=512

for i in range(1,4):

    matrix1 = tf.random.normal(shape=(n,n))
    matrix2 = tf.random.normal(shape=(n, n))
    print("Size: ", n, 'X', n)

    n = n*4

    with tf.device('/CPU:0'):
        cpu_time = timeit(lambda: matrixMultiplication(matrix1, matrix2))


    with tf.device('/GPU:0'):
        gpu_time = timeit(lambda: matrixMultiplication(matrix1, matrix2))


    print("CPU time: ", cpu_time, " seconds")
    print("GPU time: ", gpu_time," seconds")
    speedup = cpu_time / gpu_time
    print("Speedup: ", speedup)
