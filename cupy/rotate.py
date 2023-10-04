import os
os.environ["CUPY_ACCELERATORS"] = 'cub'

import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import rotate
import cv2
from timeit import timeit
from runit import runit

angle=45

def dataFunction(size):

    a = np.random.randint(0,255,size=(size,size,3),dtype=np.uint8)
    ga = cp.array(a)
    
    return a, ga

def cpuFunction(data):
    size = data.shape[0]
    return cv2.warpAffine(data,cv2.getRotationMatrix2D((size//2,size//2),angle,1),dsize=(size,size))

def gpuFunction(data):
    return rotate(data,angle,reshape=False,mode='opencv')

def gpuMemFunction(data):
    return rotate(cp.array(data),angle,reshape=False,mode='opencv').get()

runit(dataFunction, cpuFunction, gpuFunction, gpuMemFunction)
