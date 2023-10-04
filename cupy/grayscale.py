import os
os.environ["CUPY_ACCELERATORS"] = 'cub'

import numpy as np
import cupy as cp
import cv2
from timeit import timeit
from runit import runit

def dataFunction(size):

    a = np.random.uniform(0,1,size=(size,size,3)).astype(np.float32)
    ga = cp.array(a)

    c = np.array([0.299,0.587,0.114],dtype=np.float32)
    gc = cp.array(c)
    
    return (a,c), (ga,gc)

def cpuFunction(data):
    a, c = data
    return cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)

def gpuFunction(data):
    ga, gc = data
    return cp.dot(ga,gc)

def gpuMemFunction(data):
    a, c = data
    return cp.dot(cp.array(a),cp.array(c)).get()

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
        gimg = cp.array(img.astype(np.float32)/255)
        
        c = np.array([0.299,0.587,0.114],dtype=np.float32)
        gc = cp.array(c)

        cout = cpuFunction((img,c))
        gout = (255*gpuFunction((gimg,gc))).astype(np.uint8).get()

        cfile = f"{folder}/{operation_name}_opencv_{filename}"
        gfile = f"{folder}/{operation_name}_cupy_{filename}"
        
        cv2.imwrite(cfile, cout)
        cv2.imwrite(gfile, gout)

    else:
        runit(dataFunction, cpuFunction, gpuFunction, gpuMemFunction)
    
