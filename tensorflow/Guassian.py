import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image
from timeit import timeit

def gaussianFilter(image):
    filtered_image = tfa.image.gaussian_filter2d(image, filter_shape=(15, 15), sigma=15)
    filtered_image = Image.fromarray(tf.cast(filtered_image, tf.uint8).numpy())
    return filtered_image

#Apply on Actual Image
image = Image.open('/home/w374nxs/data/s-l1600.jpg')
image = tf.convert_to_tensor(image)
filtered_image = gaussianFilter(image)
filtered_image.save('/home/w374nxs/data/Guassfiltered_image.jpgg')


#Apply on Different Size
n=512

for i in range(1, 4):

    matrix = tf.random.normal((n,n))
    print("Size: ", n, 'X', n)

    n = n*4
    with tf.device('/CPU:0'):
        cpu_time = timeit(lambda: gaussianFilter(matrix))

    with tf.device('/GPU:0'):
        gpu_time = timeit(lambda: gaussianFilter(matrix))

    print("CPU time: ", cpu_time, " seconds")
    print("GPU time: ", gpu_time," seconds")
    speedup = cpu_time / gpu_time
    print("Speedup: ", speedup)
