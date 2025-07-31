from dataclasses import dataclass

from base_config import InstantiateConfig


@dataclass
class BaseSimulatorConfig(InstantiateConfig):
    module_name: str = "simulator"  # type: ignore
