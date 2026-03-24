from typing import Callable

from pacstl.core.evaluator import PacSTLEvaluator

# Registry for pacSTL evaluators, two "layers" to allow for multiple domains containing multiple rules
_REGISTRY: dict[str, dict[str, Callable[[], PacSTLEvaluator]]] = {}


def register(domain: str, rule: str):
    """Decorator: @register("colregs", "rule_8")"""

    def decorator(fn: Callable[[], PacSTLEvaluator]):
        _REGISTRY.setdefault(domain, {})[rule] = fn
        return fn

    return decorator


def create(domain: str, rule: str, **kwargs) -> PacSTLEvaluator:
    if domain not in _REGISTRY:
        raise ValueError(f"Unknown domain '{domain}'. Available: {list(_REGISTRY)}")
    rules = _REGISTRY[domain]
    if rule not in rules:
        raise ValueError(
            f"Unknown rule '{rule}' in domain '{domain}'. Available: {list(rules)}"
        )
    return rules[rule](**kwargs)


def available(domain: str | None = None) -> dict | list:
    """Introspect what's registered."""
    if domain:
        return list(_REGISTRY.get(domain, {}))
        
    return {d: list(r) for d, r in _REGISTRY.items()}
