from dataclasses import dataclass

from factory import InstantiateConfig


@dataclass
class BasePlannerConfig(InstantiateConfig):
    module_name: str = "planners"
