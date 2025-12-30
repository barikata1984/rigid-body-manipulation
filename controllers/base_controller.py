from dataclasses import dataclass

from builder import InstantiateConfig


@dataclass
class BaseControllerConfig(InstantiateConfig):
    module_name: str = "controllers"  # type: ignore
