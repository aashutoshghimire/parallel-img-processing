import os
os.environ["CUPY_ACCELERATORS"] = 'cub'

import numpy as np
import cupy as cp
import cv2
from timeit import timeit
from runit import runit

BINS = cp.arange(0,256)

def dataFunction(size):

    a = np.random.randint(0,255,size=(size,size,3),dtype=np.uint8)
    ga = cp.array(a)

    return a, ga

def cpuFunction(data):
    return np.dstack([cv2.equalizeHist(channel) for channel in np.split(data,3,axis=-1)])

def gpuFunction(data):
    himg = np.zeros_like(data)
    for c, channel in enumerate(cp.split(data,3,axis=-1)):
        hist, bins = cp.histogram(channel.flatten(),bins=256,density=True)
        cdf = hist.cumsum()
        cdf = 255 * cdf / cdf[-1]
        himg[...,c] = cp.interp(channel.flatten(), bins[:-1], cdf).reshape(data.shape[:2])
    return himg

def gpuMemFunction(data):
    return gpuFunction(cp.array(data)).get()


if __name__=="__main__":

    import sys
    import os

    process_image = False
    try:
        filepath = sys.argv[1]
        process_image = True
    except:
        pass

    if process_image:

        folder, filename = os.path.split(filepath)
        operation_name = os.path.basename(__file__).rstrip('.py')

        img = cv2.imread(filepath)

        cout = cpuFunction(img)
        gout = gpuMemFunction(img)

        cfile = f"{folder}/{operation_name}_opencv_{filename}"
        gfile = f"{folder}/{operation_name}_cupy_{filename}"

        cv2.imwrite(cfile, cout)
        cv2.imwrite(gfile, gout)

    else:
        runit(dataFunction, cpuFunction, gpuFunction, gpuMemFunction)
