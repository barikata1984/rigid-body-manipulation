from dataclasses import dataclass

from base_config import InstantiateConfig


@dataclass
class BaseControllerConfig(InstantiateConfig):
    module_name: str = "controllers"  # type: ignore
