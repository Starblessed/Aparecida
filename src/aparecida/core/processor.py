from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from PIL import Image
from torch import Tensor
from transformers import AutoProcessor


class Processor(ABC):
    def __init__(self, name: str):
        self.name: str = name
        self._processor = None

    @abstractmethod
    def preprocess_image(self, image: Image.Image) -> Any: ...

    def __call__(self, image: Image.Image) -> Any:
        return self.preprocess_image(image=image)


class SigLipProcessor(Processor):
    def __init__(self, name: str):
        super().__init__(name)
        self._processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path=self.name
        )

    def preprocess_image(self, image: Image.Image) -> Mapping[str, Tensor]:
        return self._processor(images=image, return_tensors="pt")


def get_processor(model_name: str) -> Processor:

    model_family: str = model_name.split("/")[1].split("-")[0]

    match model_family:
        case "siglip" | "siglip2":
            return SigLipProcessor(name=model_name)
        case _:
            raise ValueError(f'Unsupported model family: "{model_family}"')
