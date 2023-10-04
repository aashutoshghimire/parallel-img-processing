import os
os.environ["CUPY_ACCELERATORS"] = 'cub'

import numpy as np
import cupy as cp
import cv2
from scipy.ndimage import sobel as cpuSobel
from cupyx.scipy.ndimage import sobel as gpuSobel
from timeit import timeit
from runit import runit

def dataFunction(size):

    a = np.random.randint(0,255,size=(size,size),dtype=np.uint8)
    ga = cp.array(a)

    return a, ga

def cpuFunction(data):
    out = cpuSobel(data)
    outmin, outmax = out.min(), out.max()
    return (out-outmin)/(outmax-outmin)

def gpuFunction(data):
    out = gpuSobel(data)
    outmin, outmax = out.min(), out.max()
    return (out-outmin)/(outmax-outmin)

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

        img = cv2.cvtColor(cv2.imread(filepath),cv2.COLOR_BGR2GRAY).astype(np.float32)/255

        cout = (255*cpuFunction(img)).astype(np.uint8)
        gout = ((255*gpuMemFunction(img)).astype(np.uint8))

        cfile = f"{folder}/{operation_name}_scipy_{filename}"
        gfile = f"{folder}/{operation_name}_cupy_{filename}"

        cv2.imwrite(cfile, cout)
        cv2.imwrite(gfile, gout)

    else:
        runit(dataFunction, cpuFunction, gpuFunction, gpuMemFunction)
