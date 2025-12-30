from dataclasses import dataclass

from builder import InstantiateConfig


@dataclass
class BasePlannerConfig(InstantiateConfig):
    module_name: str = "planners"
