import interval
import numpy as np

from pacstl.common.interfaces import PACReachableSet, TimeStampedState
from pacstl.core.evaluator import PacSTLEvaluator
from pacstl.core.factory import create, register
from pacstl.core.robustness import Robustness


class SafeDistanceRobustness:
    """
    The evaluator requires a dictionary of "calculators", which are class instances
    with a compute_robustness method that translates physical states and reachable sets into robustness intervals.
    This class is a minimal example of such a calculator that computes a simple safe distance robustness.
    """

    def __init__(self, safe_radius: float):
        self.r = safe_radius

    def compute_robustness(
        self,
        ego_state: np.ndarray,
        A_matrix: np.ndarray,
        b_vector: np.ndarray,
        center: np.ndarray,
        time_step: float,
    ) -> interval.interval:
        """
        Returns an Interval [h_low, h_high] for one time step. We use the Robustness class to
        compute the lower and upper bounds of the robustness interval over the ellipsoidal reachable set.
        """
        n = A_matrix.shape[1]  # state dimension (4)
        p_ego = ego_state[:2]  # ego position

        alpha = 1.0 / self.r**2
        c = 1.0  # constant offset

        # lower bound: minimise h over the ellipsoid (SLSQP)
        # pred_Q_diag has alpha in position dims, 0 in velocity dims
        pred_Q_diag = np.zeros(n)
        pred_Q_diag[0] = alpha
        pred_Q_diag[1] = alpha

        h_low = Robustness.min_quadratic_predicates(
            pred_Q_diag=pred_Q_diag,
            pred_c=c,
            ellipsoid_A=A_matrix,
            ellipsoid_b=b_vector,
            center=center,
            x_offset=np.array([p_ego[0], p_ego[1], 0.0, 0.0]),
        )

        # upper bound: maximise h over the ellipsoid
        h_high = Robustness.max_quadratic_predicates(
            ellipsoid_A=A_matrix,
            center=center,
            dim_indices=[0, 1],  # position dimensions only
            alpha=alpha,
            c=c,
            x_offset=p_ego,
        )

        return interval.interval(float(h_low), float(h_high))


"""
We need to define the specific specification within the specific domain using 
the registry. Using the registry we store the PacSTLEvaluator instance with the 
formula and calculator for later use.
"""


@register("planar_robot", "safe_distance_globally")
def safe_distance_globally_rule(safety_radius: float = 1.0) -> PacSTLEvaluator:
    calc = SafeDistanceRobustness(safety_radius)
    return PacSTLEvaluator(
        rule_spec_string="always[0,2] (safe_distance >= 0)",
        required_signals=["safe_distance"],  # The signal name used in the formula
        calculators={"safe_distance": calc},  # The calculator instance for this signal
    )


"""
We can then use this to evaluate some hard coded trajectories and reachable sets
using the TimeStampedState and PACReachableSet dataclasses.
"""
ego_trajectory: dict[float, TimeStampedState] = {
    0.0: TimeStampedState(time_step=0.0, state_array=np.array([0.00, 0.0, 0.5, 0.0])),
    0.5: TimeStampedState(time_step=0.5, state_array=np.array([0.25, 0.0, 0.5, 0.0])),
    1.0: TimeStampedState(time_step=1.0, state_array=np.array([0.50, 0.0, 0.5, 0.0])),
    1.5: TimeStampedState(time_step=1.5, state_array=np.array([0.75, 0.0, 0.5, 0.0])),
    2.0: TimeStampedState(time_step=2.0, state_array=np.array([1.00, 0.0, 0.5, 0.0])),
}


def _ellipsoid(time_step: float, center: np.ndarray, pos_spread: float) -> PACReachableSet:
    """Helper to build one ellipsoid from a center and position spread."""
    A = np.diag([1.0 / pos_spread, 1.0 / pos_spread, 1.0 / 0.05, 1.0 / 0.05])
    return PACReachableSet(time_step, A_matrix=A, b_vector=A @ center, center=center)


reachable_tube: dict[float, PACReachableSet] = {
    #    time    centre [px,   py,  vx,   vy]    pos semi-axis (m)
    0.0: _ellipsoid(0.0, np.array([3.00, 0.8, -0.4, 0.0]), 0.30),  # spread * (1+0.0)
    0.5: _ellipsoid(0.5, np.array([2.80, 0.8, -0.4, 0.0]), 0.45),  # spread * (1+0.5)
    1.0: _ellipsoid(1.0, np.array([2.60, 0.8, -0.4, 0.0]), 0.60),  # spread * (1+1.0)
    1.5: _ellipsoid(1.5, np.array([2.40, 0.8, -0.4, 0.0]), 0.75),  # spread * (1+1.5)
    2.0: _ellipsoid(2.0, np.array([2.20, 0.8, -0.4, 0.0]), 0.90),  # spread * (1+2.0)
}


def main():
    safety_radius = 1.0

    # Create evaluator using the factory registry
    evaluator = create(
        "planar_robot", "safe_distance_globally", safety_radius=safety_radius
    )

    # Evaluate
    robustness_interval = evaluator.evaluate(ego_trajectory, reachable_tube)
    print(f"Robustness interval: {robustness_interval}")

if __name__ == "__main__":
    main()
    
