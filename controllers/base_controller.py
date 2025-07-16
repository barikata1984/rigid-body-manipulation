from dataclasses import dataclass

from configurations import InstantiateConfig


@dataclass
class BaseControllerConfig(InstantiateConfig):
    module_name: str = "controllers"  # type: ignore
