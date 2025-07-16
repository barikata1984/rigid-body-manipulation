from dataclasses import dataclass

from configurations import InstantiateConfig


@dataclass
class BasePlannerConfig(InstantiateConfig):
    module_name: str = "planners"  # type: ignore
