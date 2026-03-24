from abc import ABC

import numpy as np

from pacstl.domains.colregs.utils import (
    degree_to_radian,
    normalize_degree,
    normalize_radian_pi,
    rotation_matrix,
)


class AtomicPredicate(ABC):
    def __init__(self, factor: float, relative_to_ego: bool = True):
        super().__init__()
        self.factor = factor
        self.relative_to_ego = relative_to_ego


class InPositionHalfspace(AtomicPredicate):
    def __init__(
        self,
        degree: float,
        scaling: float,
        relative_to_ego: bool = True,
        reverse_side: bool = False,
    ):
        super().__init__(scaling, relative_to_ego)
        self.rotation_matrix = rotation_matrix(90 + degree)
        self.reverse_side_factor = -1 if reverse_side else 1

    def provide_halfspace(self, state_ego: np.ndarray) -> tuple:
        assert self.relative_to_ego
        cos_sin_ego = np.array([np.cos(state_ego[2]), np.sin(state_ego[2])])
        normal = (
            self.reverse_side_factor
            * (self.rotation_matrix @ cos_sin_ego.T)
            / self.factor
        )
        A_full = np.zeros(6)
        A_full[:2] = normal
        b = normal @ state_ego[:2]
        return A_full, b


class InFrontLeftHalfspace(InPositionHalfspace):
    def __init__(self, scaling: float, degree: float = 10):
        super().__init__(normalize_degree(degree), scaling=scaling, reverse_side=True)


class InFrontRightHalfspace(InPositionHalfspace):
    def __init__(self, scaling: float, degree: float = -10):
        super().__init__(normalize_degree(degree), scaling=scaling, reverse_side=False)


class InRightLeftHalfspace(InPositionHalfspace):
    def __init__(self, scaling: float, degree: float = -10):
        super().__init__(normalize_degree(degree), scaling=scaling, reverse_side=True)


class InRightRightHalfspace(InPositionHalfspace):
    def __init__(self, scaling: float, degree: float = -112.5):
        super().__init__(normalize_degree(degree), scaling=scaling, reverse_side=False)


class InOrientationHalfspace(AtomicPredicate):
    def __init__(
        self,
        degree: float,
        signs: np.ndarray,
        scaling: float,
        relative_to_ego: bool = True,
    ):
        super().__init__(scaling, relative_to_ego)
        self.rad = degree_to_radian(degree)
        self.signs = signs

    def provide_interval(
        self, state_ego: np.ndarray, orientation_interval: np.ndarray
    ) -> tuple:
        threshold = normalize_radian_pi(state_ego[2] + self.rad)
        lower_temp = self._check_bound(threshold, orientation_interval[0])
        upper_temp = self._check_bound(threshold, orientation_interval[1])

        # If one bound was clipped and the other not, the extremum is ±pi/2
        if lower_temp[0] == upper_temp[0] and (
            (lower_temp[2] and not upper_temp[2])
            or (not lower_temp[2] and upper_temp[2])
        ):
            if lower_temp[0] == 1:
                upper = np.pi / 2
                lower = min(
                    lower_temp[0] * lower_temp[1], upper_temp[0] * upper_temp[1]
                )
            else:
                lower = -np.pi / 2
                upper = max(
                    lower_temp[0] * lower_temp[1], upper_temp[0] * upper_temp[1]
                )
        else:
            upper = max(lower_temp[0] * lower_temp[1], upper_temp[0] * upper_temp[1])
            lower = min(lower_temp[0] * lower_temp[1], upper_temp[0] * upper_temp[1])

        return lower / self.factor, upper / self.factor

    def _check_bound(self, threshold, bound):
        diff = normalize_radian_pi(bound - threshold)
        sign = self.signs[0] if diff < 0 else self.signs[1]
        clipped = np.abs(diff) > np.pi / 2
        if clipped:
            diff = np.pi - np.abs(diff)
        return sign, np.abs(diff), clipped


class InOrientationFrontLeft(InOrientationHalfspace):
    def __init__(self, scaling: float, degree: float = 170):
        super().__init__(degree, signs=np.array([-1, 1]), scaling=scaling)


class InOrientationFrontRight(InOrientationHalfspace):
    def __init__(self, scaling: float, degree: float = -170):
        super().__init__(degree, signs=np.array([1, -1]), scaling=scaling)


class InOrientationRightLeft(InOrientationHalfspace):
    def __init__(self, scaling: float, degree: float = 170):
        super().__init__(degree, signs=np.array([1, -1]), scaling=scaling)


class InOrientationRightRight(InOrientationHalfspace):
    def __init__(self, scaling: float, degree: float = 10):
        super().__init__(degree, signs=np.array([-1, 1]), scaling=scaling)
