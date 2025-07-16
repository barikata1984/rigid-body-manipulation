from dataclasses import dataclass

from configurations import InstantiateConfig


@dataclass
class BaseSimulatorConfig(InstantiateConfig):
    moduule: str = "simulator"
