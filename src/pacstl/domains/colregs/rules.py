from pacstl.core.evaluator import PacSTLEvaluator
from pacstl.core.factory import register
from pacstl.domains.colregs.robustness_predicates import (
    InFrontLeftRobustness,
    InFrontRightRobustness,
    InOrientationFrontLeftRobustness,
    InOrientationFrontRightRobustness,
    TimeHorizonRobustness,
)
from pacstl.domains.colregs.utils import USV_DEFAULT, VesselModel, EgoVesselModel, EGO_VESSEL_DEFAULT


@register("colregs", "crossing_rule")
def _crossing_rule(vessel: VesselModel = USV_DEFAULT) -> PacSTLEvaluator:
    return PacSTLEvaluator(
        rule_spec_string="((FrontLeft >= 0) and (FrontRight >= 0) and "
        "(OriFrontLeft >= 0) and (OriFrontRight >= 0))",
        required_signals=["FrontLeft", "FrontRight", "OriFrontLeft", "OriFrontRight"],
        calculators={
            "FrontLeft": InFrontLeftRobustness(scaling=vessel.v_max),
            "FrontRight": InFrontRightRobustness(scaling=vessel.v_max),
            "OriFrontLeft": InOrientationFrontLeftRobustness(scaling=vessel.yaw_dot_max),
            "OriFrontRight": InOrientationFrontRightRobustness(scaling=vessel.yaw_dot_max),
        },
    )

@register("colregs", "crossing_detection")
def _crossing_detection(
    vessel: VesselModel = USV_DEFAULT,
    ego_vessel: EgoVesselModel = EGO_VESSEL_DEFAULT,
) -> PacSTLEvaluator:
    return PacSTLEvaluator(
        rule_spec_string=(
            "((FrontLeft >= 0) and (FrontRight >= 0) and "
            "(OriFrontLeft >= 0) and (OriFrontRight >= 0) and "
            "(collision_possible >= 0))"
        ),
        required_signals=[
            "FrontLeft", "FrontRight",
            "OriFrontLeft", "OriFrontRight",
            "collision_possible",
        ],
        calculators={
            "FrontLeft": InFrontLeftRobustness(scaling=vessel.v_max),
            "FrontRight": InFrontRightRobustness(scaling=vessel.v_max),
            "OriFrontLeft": InOrientationFrontLeftRobustness(scaling=vessel.yaw_dot_max),
            "OriFrontRight": InOrientationFrontRightRobustness(scaling=vessel.yaw_dot_max),
            "collision_possible": TimeHorizonRobustness(
                t_h=ego_vessel.t_h, a_max_ego=ego_vessel.a_max, r_ego=ego_vessel.r
            ),
        },
    )