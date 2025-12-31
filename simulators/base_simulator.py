from dataclasses import dataclass

from builder import InstantiateConfig


@dataclass
class BaseSimulatorConfig(InstantiateConfig):
    module_name: str = "simulators"  # type: ignore
