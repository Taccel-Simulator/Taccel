import torch
import os
import warp as wp
from time import perf_counter
from .log import debug
WATCH_GPU_MEM_USAGE_ENABLED = False
gpu_handle = None
pid = None

def get_gpu_mem():
    import pynvml
    global gpu_handle, pid
    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(gpu_handle)
    for process in processes:
        if process.pid == pid:
            memory_used = process.usedGpuMemory / 1024 / 1024
            return memory_used
    raise RuntimeError('The process is not on GPU; could not enable WATCH_GPU_MEM_USAGE for Profiler!')

class Profiler:

    def __init__(self, info: str, gpu_index: int=0) -> None:
        self.info = info

    def __enter__(self):
        wp.synchronize()
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        wp.synchronize()
        self.time = perf_counter() - self.start
        debug(f'[{self.info}]\t{self.time * 1000.0:.3f} ms')

class MemoryProfiler:

    def __init__(self, info: str, device: str) -> None:
        if WATCH_GPU_MEM_USAGE_ENABLED:
            import pynvml
            self.info = info
            global gpu_handle, pid
            if gpu_handle is None:
                pynvml.nvmlInit()
                pci_bus_id = wp.get_device(device).pci_bus_id
                gpu_handle = pynvml.nvmlDeviceGetHandleByPciBusId(pci_bus_id)
            if pid is None:
                pid = os.getpid()

    def __enter__(self):
        if WATCH_GPU_MEM_USAGE_ENABLED:
            wp.synchronize()
            self.start_gpu_mem = get_gpu_mem()
        return self

    def __exit__(self, type, value, traceback):
        wp.synchronize()
        if WATCH_GPU_MEM_USAGE_ENABLED:
            self.gpu_mem = get_gpu_mem() - self.start_gpu_mem
            debug(f'[{self.info}]\t{self.gpu_mem} MB')