import importlib
from dataclasses import dataclass
from typing import Any


class PrintableConfig:
    def __str__(self):
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if isinstance(val, tuple):
                flattened_val = "["
                for item in val:
                    flattened_val += str(item) + "\n"
                flattened_val = flattened_val.rstrip("\n")
                val = flattened_val + "]"
            lines += f"{key}: {str(val)}".split("\n")
        return "\n    ".join(lines)


@dataclass
class InstantiateConfig(PrintableConfig):
    module_name: str = ""
    target_class: str = ""

    def setup(self, *args, **kwargs) -> Any:
        module = importlib.import_module(self.module_name)
        target_cls = getattr(module, self.target_class)
        return target_cls(self, *args, **kwargs)


def instantiate(cfg_class: InstantiateConfig, *args, **kwargs) -> Any:
    return cfg_class.setup(*args, **kwargs)
