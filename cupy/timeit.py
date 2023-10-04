import numpy as np
import cupy as cp
from time import perf_counter

def timeit(f, gsync=False, runs=20, timeout=180):

    times = []
    total_time = 0
    for i in range(runs):
        
        if total_time > timeout:
            break

        if gsync:
            cp.cuda.Stream.null.synchronize()
        s = perf_counter()

        f()

        if gsync:
            cp.cuda.Stream.null.synchronize()
        dt = perf_counter() - s

        total_time+=dt
        times.append(dt)
    
    median_time = np.sort(times)[len(times)//2]

    return median_time
