from dataclasses import dataclass

import numpy as np


@dataclass
class TimeStampedState:
    """A generic state representation at a specific time."""

    time_step: float
    state_array: np.ndarray  


@dataclass
class PACReachableSet:
    """The mathematical definition of the uncertainty tube at one time step."""

    time_step: float
    A_matrix: np.ndarray
    b_vector: np.ndarray
    center: np.ndarray
