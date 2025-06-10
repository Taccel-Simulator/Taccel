import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import warp as wp


def set_seed(seed: int = None):
    seed = random.randint(0, 2**32 - 1) if seed is None else seed

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_robot_demo(args, demo_name: str, cache_dir: str = "./ptx/ptx_86"):
    DATETIME_TAG = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUT_DIR = f"./output/{demo_name}/{DATETIME_TAG}"
    os.makedirs(os.path.join(OUT_DIR, "frames"), exist_ok=True)
    json.dump(vars(args), open(os.path.join(OUT_DIR, "args.json"), "w"))
    set_seed(args.seed if hasattr(args, "seed") else 42)
    cache_dir = os.environ.get("TACCEL_PTX_DIR", cache_dir)
    wp.config.kernel_cache_dir = cache_dir
    wp.config.cuda_output = "ptx"
    if "TACCEL_PTX_ARCH" in os.environ:
        try:
            wp.config.ptx_target_arch = int(os.environ["TACCEL_PTX_ARCH"])
        except ValueError:
            raise ValueError("TACCEL_PTX_ARCH must be an integer (e.g., 86)")
    else:
        wp.config.ptx_target_arch = 86

    return DATETIME_TAG, OUT_DIR


def get_interp_step(start, end, n_steps, step):
    return start + (end - start) * step / n_steps
