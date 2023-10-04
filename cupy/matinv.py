import os
os.environ["CUPY_ACCELERATORS"] = 'cub'

import numpy as np
import cupy as cp
from timeit import timeit
from runit import runit

def dataFunction(size):

    a = np.random.uniform(0,1,size=(size,size))
    ga = cp.array(a)

    return a, ga

def cpuFunction(data):
    return np.linalg.inv(data)

def gpuFunction(data):
    return cp.linalg.inv(data)

def gpuMemFunction(data):
    return cp.linalg.inv(cp.array(data)).get()

runit(dataFunction, cpuFunction, gpuFunction, gpuMemFunction, runs=20)
