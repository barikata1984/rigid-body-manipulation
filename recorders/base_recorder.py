from dataclasses import dataclass

from base_config import InstantiateConfig


@dataclass
class BaseRecorderConfig(InstantiateConfig):
    module_name: str = "recorders"
