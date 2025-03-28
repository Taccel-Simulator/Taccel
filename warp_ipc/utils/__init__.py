import warp as wp
import colorful
from .profile import Profiler, MemoryProfiler

def get_val(arr: wp.array):
    wp.synchronize()
    return arr.numpy()[0]