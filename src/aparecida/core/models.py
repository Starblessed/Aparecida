from abc import ABC, abstractmethod
from typing import Any


class Model(ABC):
    """Abstract class for representing inference models.

    Attributes:
        name (str): Name of the model.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def inference(self, data) -> Any: ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.inference(*args, **kwargs)
