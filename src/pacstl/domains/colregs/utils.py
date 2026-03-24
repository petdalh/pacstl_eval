import math
from dataclasses import dataclass

import numpy as np


def rotation_matrix(angle, radian=False):
    """Create a 2D rotation matrix. Angle in degrees unless radian=True."""
    rad = angle if radian else angle * math.pi / 180
    return np.array([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]])


def normalize_degree(degree):
    """Normalize degrees to [0, 360]."""
    return degree % 360


def normalize_radian_pi(radian):
    """Normalize angle to [-pi, pi]."""
    return (radian + np.pi) % (2 * np.pi) - np.pi


def degree_to_radian(degree):
    """Convert degrees to radians."""
    return degree * math.pi / 180


@dataclass
class VesselModel:
    """Physical limits for the vessel, used to scale STL predicate robustness."""

    v_max: float = 0.4  # max speed [m/s]
    v_min: float = 0.0  # min speed [m/s]
    yaw_dot_min: float = -0.8  # min yaw rate [rad/s]
    yaw_dot_max: float = 0.8  # max yaw rate [rad/s]

@dataclass
class EgoVesselModel:
    """Physical limits for the ego vessel, used to scale STL predicate robustness."""

    a_max: float = 0.08  # max acceleration [m/s^2]
    r: float = 0.5  # safety radius [m]
    t_h: float = 20.0  # time horizon for collision prediction [s]

USV_DEFAULT = VesselModel(v_max=0.4, v_min=0.0, yaw_dot_min=-0.8, yaw_dot_max=0.8)
EGO_VESSEL_DEFAULT = EgoVesselModel(a_max=0.08, r=0.5, t_h=20.0)
