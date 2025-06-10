import colorful
import warp as wp
from .profile import MemoryProfiler as MemoryProfiler, Profiler as Profiler

def get_val(arr: wp.array):
    wp.synchronize()
    return arr.numpy()[0]