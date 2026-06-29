import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any



# Pretty printing class
class PrintableConfig:
    """Printable Config defining str function"""

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


# Base instantiate configs
@dataclass
class InstantiateConfig(ABC, PrintableConfig):
    """Config class for instantiating an the class specified in the _target attribute."""

    @property
    @abstractmethod
    def module_name(self) -> str: ...

    @property
    @abstractmethod
    def target_class(self) -> str: ...

    #def setup(self, m, d, *args, **kwargs) -> Any:
    def setup(self, *args, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        module = importlib.import_module(self.module_name)
        target_class = getattr(module, self.target_class)

        #return target_class(self, m, d, *args, **kwargs)
        return target_class(self, *args, **kwargs)


#def instantiate(cfg_class: Any, m: MjModel, d: MjData, *args, **kwargs) -> Any:
#    # cfg_class = OmegaConf.to_object(cfg)
#    return cfg_class.setup(m, d, *args, **kwargs)  # type: ignore
def instantiate(cfg_class: Any, *args, **kwargs) -> Any:
    # cfg_class = OmegaConf.to_object(cfg)
    return cfg_class.setup(*args, **kwargs)  # type: ignore

