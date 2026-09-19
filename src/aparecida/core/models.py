from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Literal

from torch import Tensor
from transformers import Siglip2VisionModel, SiglipVisionModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

SUPPORTED_MODEL = Literal["google/siglip-so400m-patch14-384"]
SUPPORTED_MODELS = ["google/siglip-so400m-patch14-384"]


class Model(ABC):
    """Abstract class for representing inference models.

    Attributes:
        name (str): Name of the model.
    """

    def __init__(self, name: str):
        self.name = name
        self._predictor = None

    @abstractmethod
    def inference(self, data) -> Any: ...

    def __call__(self, data) -> Any:
        return self.inference(data=data)


class SigLipModel(Model):
    def __init__(self, name: str):
        super().__init__(name=name)
        self._predictor = SiglipVisionModel.from_pretrained(self.name)

    def inference(self, data: Mapping[str, Tensor]) -> BaseModelOutputWithPooling:
        return self._predictor(**data)


class SigLip2Model(Model):
    def __init__(self, name: str):
        super().__init__(name=name)
        self._predictor = Siglip2VisionModel.from_pretrained(self.name)

    def inference(self, data: Mapping[str, Tensor]) -> BaseModelOutputWithPooling:
        return self._predictor(**data)


def get_model(model_name: SUPPORTED_MODEL) -> Model:
    if model_name not in SUPPORTED_MODELS:
        raise TypeError(f'Unsupported model name: "{model_name}"')

    model_family: str = model_name.split("/")[1].split("-")[0]

    match model_family:
        case "siglip":
            return SigLipModel(model_name)
        case "siglip2":
            return SigLip2Model(model_name)
        case _:
            raise ValueError(f'Unsupported model family: "{model_family}"')
