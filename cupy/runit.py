from timeit import timeit

sizes = (512, 2048, 8192)

def runit(dataFunction, cpuFunction, gpuFunction, gpuMemFunction, runs=100):

    for size in sizes:
        
        print(f"size: {size}x{size}")

        cpuData, gpuData = dataFunction(size)
        
        cpuTime = timeit(lambda: cpuFunction(cpuData), runs=runs)
        print(f"cpu: {cpuTime:.3e}")
    
        gpuTime = timeit(lambda: gpuFunction(gpuData), gsync=True, runs=runs)
        print(f"gpu: {gpuTime:.3e}")
        print(f"operation speedup: {cpuTime/gpuTime:.2f}")
        
        gpuMemTime = timeit(lambda: gpuMemFunction(cpuData), gsync=True, runs=runs)
        print(f"gpu+mem: {gpuMemTime:.3e}")
        print(f"total speedup: {cpuTime/gpuMemTime:.2f}")
