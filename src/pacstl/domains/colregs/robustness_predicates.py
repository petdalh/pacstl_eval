import interval
import numpy as np

from pacstl.core.robustness import Robustness
from pacstl.domains.colregs.predicates import (
    InFrontLeftHalfspace,
    InFrontRightHalfspace,
    InOrientationFrontLeft,
    InOrientationFrontRight,
    InOrientationRightLeft,
    InOrientationRightRight,
    InRightLeftHalfspace,
    InRightRightHalfspace,
)


class BaseHalfspaceRobustness(Robustness):
    """Base class for computing robustness using halfspace predicates."""

    def compute_robustness(
        self, state_ego_array, ellipsoid_A, ellipsoid_b, center, time_step
    ):
        pred_A, pred_b = self.pred.provide_halfspace(state_ego_array)
        min_h = self.min_linear_predicates(
            pred_A, pred_b, ellipsoid_A, ellipsoid_b, center
        )
        max_h = self.max_linear_predicates(
            pred_A, pred_b, ellipsoid_A, ellipsoid_b, center
        )
        return interval.interval(min(min_h, max_h), max(min_h, max_h))

    def __call__(self, state_ego_array, ellipsoid_A, ellipsoid_b, center, time_step):
        return self.compute_robustness(
            state_ego_array, ellipsoid_A, ellipsoid_b, center, time_step
        )


class BaseOrientationRobustness(Robustness):
    """Base class for computing robustness using orientation predicates."""

    def __init__(
        self, pred, obstacle_state_dim: int = 6, obstacle_orientation_idx: int = 2
    ):
        super().__init__(pred=pred)
        # Dynamically build the extraction vector
        self._obstacle_orientation_vector = np.zeros(obstacle_state_dim)
        self._obstacle_orientation_vector[obstacle_orientation_idx] = 1.0
        # Extracted constant to prevent re-allocating memory on every computation

    def compute_robustness(
        self, state_ego_array, ellipsoid_A, ellipsoid_b, center, time_step
    ):
        min_ori = self.min_linear_predicates(
            self._obstacle_orientation_vector, 0.0, ellipsoid_A, ellipsoid_b, center
        )
        max_ori = self.max_linear_predicates(
            self._obstacle_orientation_vector, 0.0, ellipsoid_A, ellipsoid_b, center
        )
        orientation_interval = np.array([min(min_ori, max_ori), max(min_ori, max_ori)])
        min_h, max_h = self.pred.provide_interval(state_ego_array, orientation_interval)
        return interval.interval(min(min_h, max_h), max(min_h, max_h))

    def __call__(self, state_ego_array, ellipsoid_A, ellipsoid_b, center, time_step):
        return self.compute_robustness(
            state_ego_array, ellipsoid_A, ellipsoid_b, center, time_step
        )


# --- Specific Implementations ---


class InFrontLeftRobustness(BaseHalfspaceRobustness):
    def __init__(self, scaling: float):
        super().__init__(pred=InFrontLeftHalfspace(scaling=scaling))


class InFrontRightRobustness(BaseHalfspaceRobustness):
    def __init__(self, scaling: float):
        super().__init__(pred=InFrontRightHalfspace(scaling=scaling))


class InRightLeftRobustness(BaseHalfspaceRobustness):
    def __init__(self, scaling: float):
        super().__init__(pred=InRightLeftHalfspace(scaling=scaling))


class InRightRightRobustness(BaseHalfspaceRobustness):
    def __init__(self, scaling: float):
        super().__init__(pred=InRightRightHalfspace(scaling=scaling))


class InOrientationFrontRightRobustness(BaseOrientationRobustness):
    def __init__(self, scaling: float):
        super().__init__(pred=InOrientationFrontRight(scaling=scaling))


class InOrientationFrontLeftRobustness(BaseOrientationRobustness):
    def __init__(self, scaling: float):
        super().__init__(pred=InOrientationFrontLeft(scaling=scaling))


class InOrientationRightRightRobustness(BaseOrientationRobustness):
    def __init__(self, scaling: float):
        super().__init__(pred=InOrientationRightRight(scaling=scaling))


class InOrientationRightLeftRobustness(BaseOrientationRobustness):
    def __init__(self, scaling: float):
        super().__init__(pred=InOrientationRightLeft(scaling=scaling))


# --- Timed Robustness Predicates ---
class TimeHorizonRobustness(Robustness):
    """Robustness for the time_horizon atomic proposition (Eq. 35 in the paper).

    Computes whether there is a collision risk within a time horizon by
    comparing relative speed to normalized relative distance.
    """

    def __init__(
        self,
        t_h: float = 5.0,
        a_max_ego: float = 0.08,
        r_ego: float = 0.5,
        pos_indices: list = None,
        vel_indices: list = None,
    ):
        super().__init__(pred=None)
        self.a_max_ego = a_max_ego
        self.t_h = t_h
        self.r_ego = r_ego
        self.pos_indices = pos_indices or [0, 1]
        self.vel_indices = vel_indices or [3, 4]

    def compute_robustness(
        self, state_ego_array, ellipsoid_A, ellipsoid_b, center, time_step
    ):
        p_ego = state_ego_array[self.pos_indices]
        v_ego = state_ego_array[self.vel_indices]

        n = ellipsoid_A.shape[1]
        Q_diag_pos = np.zeros(n)
        Q_diag_pos[self.pos_indices] = 1.0
        x_offset_pos = np.zeros(n)
        x_offset_pos[self.pos_indices] = p_ego

        # This gives min ||p_other - p_ego||^2 over the ellipsoid
        min_dist_sq = self.min_quadratic_predicates(
            Q_diag_pos, 0.0, ellipsoid_A, ellipsoid_b, center, x_offset=x_offset_pos
        )
        gamma_min = max(np.sqrt(max(min_dist_sq, 0.0)) - self.r_ego, 0.0)

        x_offset_vel = np.zeros(n)
        x_offset_vel[self.vel_indices] = v_ego
        A_inv = np.linalg.inv(ellipsoid_A)
        center_ell = A_inv @ ellipsoid_b

        max_vrel_sq = self.max_quadratic_predicates(
            ellipsoid_A, center_ell, self.vel_indices, alpha=1.0, c=0.0,
            x_offset=v_ego
        )
        max_vrel = np.sqrt(max(max_vrel_sq, 0.0))

        h_low = (1.0 / self.a_max_ego) * (max_vrel - gamma_min / self.t_h)

        max_dist_sq = self.max_quadratic_predicates(
            ellipsoid_A, center_ell, self.pos_indices, alpha=1.0, c=0.0,
            x_offset=p_ego
        )
        gamma_max = max(np.sqrt(max(max_dist_sq, 0.0)) - self.r_ego, 0.0)

        min_vrel_sq = self.min_quadratic_predicates(
            Q_diag_pos * 0.0 + np.array([
                1.0 if i in self.vel_indices else 0.0 for i in range(n)
            ]),
            0.0, ellipsoid_A, ellipsoid_b, center, x_offset=x_offset_vel
        )
        min_vrel = np.sqrt(max(min_vrel_sq, 0.0))

        h_high = (1.0 / self.a_max_ego) * (min_vrel - gamma_max / self.t_h)

        return interval.interval(min(h_low, h_high), max(h_low, h_high))