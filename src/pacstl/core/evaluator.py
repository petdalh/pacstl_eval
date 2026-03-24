import rtamt


class PacSTLEvaluator:
    def __init__(
        self, rule_spec_string: str, required_signals: list, calculators: dict
    ):
        # Setup the RTAMT spec engine
        self.spec = rtamt.StlDiscreteTimeSpecification(
            semantics=rtamt.Semantics.PAC_STL
        )
        self.required_signals = required_signals
        self.calculators = calculators

        # Declare variables as intervals for pacSTL
        for sig in self.required_signals:
            self.spec.declare_var(sig, "interval")

        self.spec.spec = rule_spec_string
        self.spec.parse()

    def evaluate(self, ego_trajectory: dict, reachable_tube: dict) -> float:
        """
        Translates pre-computed trajectories/tubes into an RTAMT trace,
        evaluates it, and returns the robustness.
        """
        # Initialize the trace format RTAMT expects
        trace = {"time": [], **{sig: [] for sig in self.required_signals}}

        # Calculate predicates for overlapping timesteps
        for time_step, pac_set in reachable_tube.items():
            if time_step not in ego_trajectory:
                continue

            ego_state = ego_trajectory[time_step]
            trace["time"].append(time_step)

            # Convert physical states into robustness intervals
            for sig, calc in self.calculators.items():
                result = calc.compute_robustness(
                    ego_state.state_array,
                    pac_set.A_matrix,
                    pac_set.b_vector,
                    pac_set.center,
                    time_step,
                )
                trace[sig].append(result)  # Append the interval

        # Evaluate using RTAMT 
        return self.spec.evaluate(trace)
