from dataclasses import dataclass

from configurations import InstantiateConfig


@dataclass
class BaseRecorderConfig(InstantiateConfig):
    module_name: str = "recorders"
