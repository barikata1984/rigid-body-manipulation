from dataclasses import dataclass

from factory import InstantiateConfig


@dataclass
class BaseControllerConfig(InstantiateConfig):
    module_name: str = "controllers"
