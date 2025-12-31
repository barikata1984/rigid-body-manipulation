from dataclasses import dataclass

from factory import InstantiateConfig


@dataclass
class BaseSimulatorConfig(InstantiateConfig):
    module_name: str = "simulators"  # type: ignore
