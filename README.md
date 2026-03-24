# About

pacstl_evaluator is a python package for setting up and evaluating pacSTL. pacSTL is a framework that combines Probably Approximately Correct (PAC) bounded set predictions with an interval extension of Signal Temporal Logic (STL).

# Project Structure

```
pacstl_evaluator/
├── pyproject.toml # Project config (setuptools, v0.1.0)
├── README.md
├── requirements.txt
└── src/
    └── pacstl/ # Main package
        ├── __init__.py
        ├── main.py            
        ├── common/
        │   ├── __init__.py
        │   └── interfaces.py  
        ├── core/
        │   ├── __init__.py
        │   ├── evaluator.py  # Main evaluation class
        │   ├── factory.py    # Rule registry 
        │   └── robustness.py # Robustness over ellipsoidal reachable sets
        └── domains/
            ├── __init__.py
            └── colregs/ # COLREGs (maritime traffic rules) domain
                ├── __init__.py
                ├── predicates.py # Halfspace & orientation predicates
                ├── robustness_predicates.py # Robustness implementations
                ├── rules.py # COLREGs rule definitions
                └── utils.py # VesselModel, rotation utilities
```

## Key Components

| Module                                     | Description                                                                                                         |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| `common/interfaces.py`                     | `TimeStampedState` (time + state array) and `PACReachableSet` (ellipsoid center, A-matrix, b-vector)                |
| `core/evaluator.py`                        | `PacSTLEvaluator` — parses STL spec, computes interval-valued robustness signals via RTAMT with `PAC_STL` semantics |
| `core/factory.py`                          | Two-level registry (`domain → rule`); decorator-based rule registration                                             |
| `core/robustness.py`                       | Min/max optimization of linear and quadratic predicates over ellipsoids               |
| `domains/colregs/predicates.py`            | Halfspace and orientation `AtomicPredicate` subclasses for spatial COLREGs constraints                              |
| `domains/colregs/robustness_predicates.py` | Interval robustness over ellipsoids for each predicate type                                                         |
| `domains/colregs/rules.py`                 | rules registered under `"colregs"` domain                                                         |
| `domains/colregs/utils.py`                 | `VesselModel` dataclass containing `USV_DEFAULT` and `EGO_VESSEL_DEFAULT`                                       |

## Dependencies

| Dependency                                                                            | Purpose                       | Source / Version                             |
| :------------------------------------------------------------------------------------ | :---------------------------- | :------------------------------------------- |
| **[rtamt](https://github.com/petdalh/rtamt/tree/istl_implementation_antlr4_upgrade)** | STL/PAC-STL evaluation engine | `branch: istl_implementation_antlr4_upgrade` |
| **[npinterval](https://github.com/petdalh/npinterval)**                               | Interval arithmetic           | GitHub Repository                            |
| **NumPy**                                                                             | Numerical computations        | Standard (PyPI)                              |
| **SciPy**                                                                             | SLSQP & eigenvalue solvers    | Standard (PyPI)                              |
## Example Usage: Planar Robot

The repository includes a COLREGs (maritime traffic rules) domain implementation, but the core pacSTL functionality is domain-agnostic. This section walks through extending the package to a minimal new domain: a planar robot monitoring whether another agent stays at a safe distance. A full runnable example can be found in `example.py`

#### 1. Define a robustness calculator

Each atomic predicate needs a calculator class with a `compute_robustness` method. This method receives the ego state and the ellipsoidal reachable set parameters at a single time step, and returns an interval `[h_low, h_high]` by optimizing over the ellipsoid using the `Robustness` class.
```Python
class SafeDistanceRobustness:
    def __init__(self, safe_radius: float):
        self.r = safe_radius

    def compute_robustness(self, ego_state: np.ndarray, A_matrix: np.ndarray, b_vector: np.ndarray, center: np.ndarray, time_step: float) -> interval.interval:
        n = A_matrix.shape[1]
        p_ego = ego_state[:2]
        alpha = 1.0 / self.r**2
        c = 1.0
        pred_Q_diag = np.zeros(n)
        pred_Q_diag[0] = alpha
        pred_Q_diag[1] = alpha
        h_low = Robustness.min_quadratic_predicates(pred_Q_diag=pred_Q_diag, pred_c=c, ellipsoid_A=A_matrix, ellipsoid_b=b_vector, center=center, x_offset=np.array([p_ego[0], p_ego[1], 0.0, 0.0]))
        h_high = Robustness.max_quadratic_predicates(ellipsoid_A=A_matrix, center=center, dim_indices=[0, 1], alpha=alpha, c=c, x_offset=p_ego)
        return interval.interval(float(h_low), float(h_high))
```

#### 2. Register a rule with the factory

Use the `@register` decorator to bind a domain name and rule name to a factory function. The factory returns a configured `PacSTLEvaluator` with the STL specification string, the signal names (atomic propositions), and the calculator instances from step 1.
```Python
@register("planar_robot", "safe_distance_globally")
def safe_distance_globally_rule(safety_radius: float = 1.0) -> PacSTLEvaluator:
	calc = SafeDistanceRobustness(safety_radius)
	return PacSTLEvaluator(
		rule_spec_string="always[0,2] (safe_distance >= 0)",
		required_signals=["safe_distance"],
		calculators={"safe_distance": calc},
	)
```

#### 3. Evaluate

Create an evaluator through the factory and call `evaluate` with an ego trajectory and a reachable tube. Both are dictionaries keyed by time step, containing `TimeStampedState` and `PACReachableSet` instances respectively. `example.py`uses a hard coded reachable tube and ego trajectory. 

```Python
evaluator = create("planar_robot", "safe_distance_globally", safety_radius=safety_radius)
robustness_interval = evaluator.evaluate(ego_trajectory, reachable_tube)
```

