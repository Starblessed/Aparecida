from collections.abc import Mapping
from secrets import token_hex

from PIL import Image
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPooling

from aparecida.core.models import SUPPORTED_MODEL, Model, get_model
from aparecida.core.processor import Processor, get_processor
from aparecida.utils.logger import get_logger

LOGGER = get_logger("Engine")


class Engine:
    def __init__(self, model_name: SUPPORTED_MODEL):
        self.model_name: SUPPORTED_MODEL = model_name
        self.id: str = token_hex(3) + "-" + token_hex(3)

    def initialize(self):
        LOGGER.info(f"Initializing engine {self.id}...")

        self.processor: Processor = get_processor(model_name=self.model_name)
        self.model: Model = get_model(model_name=self.model_name)

        LOGGER.info(f"Engine {self.id} ready.")

    def encode(self, inputs: Mapping[str, Tensor]) -> BaseModelOutputWithPooling:
        LOGGER.info(f"Encoding with engine {self.id}...")
        outputs: BaseModelOutputWithPooling = self.model(data=inputs)
        LOGGER.info("Encoding done.")

        return outputs

    def preprocess_image(self, image: Image.Image):
        LOGGER.info(f"Preprocessing with engine {self.id}...")
        return self.processor(image=image)
