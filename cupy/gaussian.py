import os
os.environ["CUPY_ACCELERATORS"] = 'cub'

import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter
import cv2
from timeit import timeit
from runit import runit

k_size = 15

def dataFunction(size):

    a = np.random.uniform(0,1,size=(size,size)).astype(np.float32)
    ga = cp.array(a)
    
    return a, ga

def cpuFunction(data):
    return cv2.GaussianBlur(data,(k_size,k_size),sigmaX=k_size,sigmaY=k_size)

def gpuFunction(data):
    sigmas = [k_size//2 if i < 2 else 0 for i in range(len(data.shape))]
    return gaussian_filter(data,sigmas)

def gpuMemFunction(data):
    sigmas = [k_size//2 if i < 2 else 0 for i in range(len(data.shape))]
    return gaussian_filter(cp.array(data),sigmas).get()


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

        img = cv2.cvtColor(cv2.imread(filepath),cv2.COLOR_BGR2RGB)
        gimg = cp.array(img).astype(np.float32)/255

        cout = cpuFunction(img)
        gout = (255*gpuFunction(gimg)).astype(np.uint8).get()

        cfile = f"{folder}/{operation_name}_opencv_{filename}"
        gfile = f"{folder}/{operation_name}_cupy_{filename}"

        cv2.imwrite(cfile, cv2.cvtColor(cout,cv2.COLOR_RGB2BGR))
        cv2.imwrite(gfile, cv2.cvtColor(gout,cv2.COLOR_RGB2BGR))

    else:
        runit(dataFunction, cpuFunction, gpuFunction, gpuMemFunction)
