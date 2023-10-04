import os
os.environ["CUPY_ACCELERATORS"] = 'cub'

import numpy as np
import cupy as cp
from timeit import timeit
from runit import runit

def dataFunction(size):

    a = np.random.uniform(0,1,size=(size,size))
    b = np.random.uniform(0,1,size=(size,size))
    ga = cp.array(a)
    gb = cp.array(b)

    return (a,b), (ga,gb)

def cpuFunction(data):
    a, b = data
    return np.matmul(a,b)

def gpuFunction(data):
    ga, gb = data
    return cp.matmul(ga,gb)

def gpuMemFunction(data):
    a, b = data
    return cp.matmul(cp.array(a),cp.array(b)).get()

runit(dataFunction, cpuFunction, gpuFunction, gpuMemFunction, runs=20)
