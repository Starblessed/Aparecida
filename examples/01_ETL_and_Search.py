"""
from src.infra.vector.data_loader import Loader
from src.infra.vector.manager import VectorDatabaseManager


from src.core.postprocessing import embedding_to_normalized_list
"""

import hashlib
import random
from pathlib import Path

from PIL import Image

from aparecida.core.engine import Engine
from aparecida.core.loader import load_image


def image_sha256(image: Image.Image) -> str:
    h = hashlib.sha256()
    h.update(image.width.to_bytes(4, "big"))
    h.update(image.height.to_bytes(4, "big"))
    h.update(image.tobytes())
    return h.hexdigest()


def sample_n(data: list, n: int):
    data = data.copy()
    samples: list = [data.pop(data.index(random.choice(data))) for k in range(n)]
    return samples


MODEL_NAME: str = "google/siglip-so400m-patch14-384"
DB_NAME: str = "example_01"

N_DB_SAMPLES: int = 4

INPUT_DATA_PATH: Path = Path("input") / "microsoft-digiface-1m-sample"

if __name__ == "__main__":
    # 1 --------------- Load Data

    data: dict = {"entries": []}

    identities: list[Path] = list(INPUT_DATA_PATH.iterdir())

    for identity in identities:
        images: list[Path] = list(identity.iterdir())
        images = sample_n(images, n=N_DB_SAMPLES)

        for image in images:
            image_data = load_image(source=image)
            data["entries"].append(
                {
                    "id": image_sha256(image=image_data),
                    "name": identity.name,
                    "vector": None,
                    "source": "ms-digiface-1m",
                    "path": image,
                }
            )

    print(data)

    # 2 --------------- Load Model
    engine: Engine = Engine(model_name=MODEL_NAME)
    # TODO: Model Loading Pipeline
