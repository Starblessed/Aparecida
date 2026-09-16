from collections.abc import Mapping
from typing import Literal

from PIL import Image
from torch import Tensor
from transformers import AutoProcessor, Siglip2VisionModel, SiglipVisionModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

SUPPORTED_MODEL = Literal["google/siglip-so400m-patch14-384"]
SUPPORTED_MODELS = ["google/siglip-so400m-patch14-384"]


class Engine:
    def __init__(self, model_name: SUPPORTED_MODEL):
        self.model_name = model_name

    def initialize(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model: SiglipVisionModel | Siglip2VisionModel = self.__get_model()

    def __get_model(self):
        if self.model_name not in SUPPORTED_MODELS:
            raise TypeError(f'Unsupported model: "{self.model_name}"')

        model_family: str = self.model_name.split("/")[1].split("-")[0]

        match model_family:
            case "siglip":
                return SiglipVisionModel.from_pretrained(self.model_name)
            case "siglip2":
                return Siglip2VisionModel.from_pretrained(self.model_name)
            case _:
                raise ValueError(f'Unsupported model family: "{model_family}"')

    def inference(self, inputs: Mapping[str, Tensor]) -> BaseModelOutputWithPooling:
        outputs: BaseModelOutputWithPooling = self.model(**inputs)

        return outputs

    def preprocess_image(self, image: Image.Image):
        return self.processor(images=image, return_tensors="pt")
