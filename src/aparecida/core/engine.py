from collections.abc import Mapping
from secrets import token_hex
from typing import Literal

from PIL import Image
from torch import Tensor
from transformers import AutoProcessor, Siglip2VisionModel, SiglipVisionModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from aparecida.utils.logger import get_logger

SUPPORTED_MODEL = Literal["google/siglip-so400m-patch14-384"]
SUPPORTED_MODELS = ["google/siglip-so400m-patch14-384"]

LOGGER = get_logger("Engine")


class Engine:
    def __init__(self, model_name: SUPPORTED_MODEL):
        self.model_name: str = model_name
        self.id: str = token_hex(3) + "-" + token_hex(3)

    def initialize(self):
        LOGGER.info(f"Initializing engine {self.id}...")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model: SiglipVisionModel | Siglip2VisionModel = self.__get_model()
        LOGGER.info(f"Engine {self.id} ready.")

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

    def encode(self, inputs: Mapping[str, Tensor]) -> BaseModelOutputWithPooling:
        LOGGER.info(f"Encoding with engine {self.id}...")
        outputs: BaseModelOutputWithPooling = self.model(**inputs)
        LOGGER.info("Encoding done.")

        return outputs

    def preprocess_image(self, image: Image.Image):
        return self.processor(images=image, return_tensors="pt")
