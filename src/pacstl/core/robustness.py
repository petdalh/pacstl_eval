import numpy as np
import scipy.optimize
from scipy.optimize import NonlinearConstraint, brentq


class Robustness:
    """Calculate robustness functions over ellipsoidal reachable sets."""

    def __init__(self, pred=None):
        self.pred = pred

    @staticmethod
    def min_linear_predicates(pred_A, pred_b, ellipsoid_A, ellipsoid_b, init):
        """Minimize a linear predicate pred_A^T p - pred_b over an ellipsoid."""

        def constr_fun(p):
            diff = ellipsoid_A @ p - ellipsoid_b
            return np.dot(diff, diff) - 1

        def constr_jac(p):
            return 2 * (ellipsoid_A.T @ (ellipsoid_A @ p - ellipsoid_b))

        in_set = NonlinearConstraint(constr_fun, -np.inf, 0, jac=constr_jac)
        h_low = scipy.optimize.minimize(
            lambda p: pred_A.T @ p - pred_b,
            init,
            jac=lambda p: pred_A,
            constraints=in_set,
            method="SLSQP",
            options={"maxiter": 200},
        )
        if not h_low.success:
            print("Opt not term")
            print(h_low)
            return None
        return h_low.fun

    @staticmethod
    def max_linear_predicates(pred_A, pred_b, ellipsoid_A, ellipsoid_b, init):
        """Maximize a linear predicate pred_A^T p - pred_b over an ellipsoid."""

        def constr_fun(p):
            diff = ellipsoid_A @ p - ellipsoid_b
            return np.dot(diff, diff) - 1

        def constr_jac(p):
            return 2 * (ellipsoid_A.T @ (ellipsoid_A @ p - ellipsoid_b))

        in_set = NonlinearConstraint(constr_fun, -np.inf, 0, jac=constr_jac)
        h_high = scipy.optimize.minimize(
            lambda p: -(pred_A.T @ p - pred_b),
            init,
            jac=lambda p: -pred_A,
            constraints=in_set,
            method="SLSQP",
            options={"maxiter": 200},
        )
        if not h_high.success:
            print("Opt not term")
            print(h_high)
            return None
        return -h_high.fun

    @staticmethod
    def min_quadratic_predicates(
        pred_Q_diag, pred_c, ellipsoid_A, ellipsoid_b, center, x_offset=None
    ):
        """Minimize (p - x_offset)^T diag(pred_Q_diag) (p - x_offset) - pred_c over an ellipsoid."""
        n = ellipsoid_A.shape[1]
        if x_offset is None:
            x_offset = np.zeros(n)

        def objective(p):
            diff = p - x_offset
            return np.sum(pred_Q_diag * (diff**2)) - pred_c

        cons = NonlinearConstraint(
            lambda p: np.linalg.norm(ellipsoid_A @ p - ellipsoid_b), -np.inf, 1.0
        )
        res = scipy.optimize.minimize(
            objective,
            center,
            method="SLSQP",
            constraints=[cons],
            options={"ftol": 1e-9, "disp": False},
        )
        return res.fun

    @staticmethod
    def max_quadratic_predicates(
        ellipsoid_A, center, dim_indices, alpha, c, x_offset=None
    ):
        """Analytically maximize alpha * ||x||^2 - c where x is a projection of the ellipsoid.

        Uses a Lagrange multiplier / secular equation approach (Brent's method) — no
        numerical optimization required.

        Parameters
        ----------
        ellipsoid_A : ndarray
            Matrix defining the ellipsoid ||A v - b|| <= 1.
        center : ndarray
            Center of the ellipsoid (A^-1 b).
        dim_indices : list of int
            Dimensions to project onto (e.g. [0, 1] for position, [3, 4] for velocity).
        alpha : float
            Positive scalar weight on the quadratic term.
        c : float
            Constant offset subtracted from the objective.
        x_offset : ndarray, optional
            Shift applied before computing the norm (defaults to zero).
        """
        A_inv = np.linalg.inv(ellipsoid_A)
        Sigma_full = A_inv @ A_inv.T

        idx = np.ix_(dim_indices, dim_indices)
        Sigma_x = Sigma_full[idx]
        x_c = center[dim_indices]

        lambdas, U = np.linalg.eigh(Sigma_x)

        d = U.T @ x_c if x_offset is None else U.T @ (x_c - x_offset)
        d = np.where(np.abs(d) < 1e-12, 1e-12, d)

        def secular_eq(gamma):
            return np.sum(lambdas * (d**2) / (gamma - lambdas) ** 2) - 1.0

        gamma_max = np.max(lambdas)
        low = gamma_max + 1e-11
        high = gamma_max + np.sqrt(np.sum(lambdas * (d**2))) + 1.0

        while secular_eq(low) < 0 and low > gamma_max:
            low = gamma_max + (low - gamma_max) / 10.0

        try:
            gamma_opt = brentq(secular_eq, low, high)
        except ValueError:
            gamma_opt = low

        v_opt = U @ ((gamma_opt / (gamma_opt - lambdas)) * d)
        return alpha * np.linalg.norm(v_opt) ** 2 - c
