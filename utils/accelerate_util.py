import os
import random

import numpy as np
import torch
from loguru import logger


def accelerator_load_random_state(ckpt_dir: str, accelerator, process_index: int = 0):
    """ Load random states from a file and set the random seeds for various libraries.
        No idea if this will work for other process_index than 0.
    """

    try:
        states = torch.load(os.path.join(ckpt_dir, f"random_states_{process_index}.pkl"), weights_only=False)
        if "step" in states:
            accelerator.step = states["step"]
        random.setstate(states["random_state"])
        np.random.set_state(states["numpy_random_seed"])
        torch.set_rng_state(states["torch_manual_seed"])
        if torch.cuda.is_available() and "torch_cuda_manual_seed" in states:
            torch.cuda.set_rng_state_all(states["torch_cuda_manual_seed"])
    except Exception as e:
        logger.error(f"Encountered an issue when loading the accelerator random state. Exception raised: {e}")
