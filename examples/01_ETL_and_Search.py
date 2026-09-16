import hashlib
import random
from pathlib import Path

from PIL import Image

from aparecida.core.engine import Engine
from aparecida.core.loader import load_image
from aparecida.core.postprocessing import embedding_to_normalized_list
from aparecida.infra.vector.data_loader import LANCE_SCHEMA
from aparecida.infra.vector.manager import VectorDatabaseManager
from aparecida.utils.logger import get_logger


def image_sha256(image: Image.Image) -> str:
    h = hashlib.sha256()
    h.update(image.width.to_bytes(4, "big"))
    h.update(image.height.to_bytes(4, "big"))
    h.update(image.tobytes())
    return h.hexdigest()


def sample_n_then_k(data: list, n: int, k: int):
    data = data.copy()

    samples_n: list = [data.pop(data.index(random.choice(data))) for _ in range(n)]
    samples_k: list = [data.pop(data.index(random.choice(data))) for _ in range(k)]

    return samples_n, samples_k


MODEL_NAME: str = "google/siglip-so400m-patch14-384"
DB_NAME: str = "example_01"
DB_PATH: str = str(Path("output") / DB_NAME)
TABLE_NAME: str = "persons"

N_DB_SAMPLES: int = 2
N_SEARCHES_PER_ID: int = 1  # Do not change

INPUT_DATA_PATH: Path = Path("input") / "microsoft-digiface-1m-sample"

if __name__ == "__main__":
    logger = get_logger("Pipeline")

    # 1 --------------- Load Data

    data: dict[str, dict] = {}

    indices: dict[str, dict] = {}

    identities: list[Path] = list(INPUT_DATA_PATH.iterdir())

    for identity in identities:
        images: list[Path] = list(identity.iterdir())
        corpus, index = sample_n_then_k(images, n=N_DB_SAMPLES, k=N_SEARCHES_PER_ID)

        for image in corpus:
            image_data = load_image(source=image)
            img_hash = image_sha256(image=image_data)
            data.update(
                {
                    img_hash: {
                        "id": img_hash,
                        "name": identity.name,
                        "vector": None,
                        "source": "ms-digiface-1m",
                        "path": image,
                    }
                }
            )

        for image in index:
            image_data = load_image(source=image)
            img_hash = image_sha256(image=image_data)
            indices.update(
                {
                    identity.name: {
                        "name": identity.name,
                        "vector": None,
                        "path": image,
                        "matches": [],
                        "correct_match": False,
                    }
                }
            )

    # 2 --------------- Load Model
    engine: Engine = Engine(model_name=MODEL_NAME)
    engine.initialize()

    # 3 --------------- Load Vector DB
    db_manager = VectorDatabaseManager(uri=DB_PATH)
    db_manager.start_table(
        name=TABLE_NAME, schema=LANCE_SCHEMA, overwrite_existing=True
    )

    corpus_size: int = len(data)

    for i, (id, entry) in enumerate(data.items()):
        image = load_image(entry["path"])
        inputs = engine.preprocess_image(image)

        logger.info(f"Encoding entry {entry['id']} ({i + 1} of {corpus_size})...")

        outputs = engine.encode(inputs=inputs)

        vector = embedding_to_normalized_list(outputs)

        data[id]["vector"] = vector
        data[id].pop("path", None)

    # 4 --------------- Postprocess Data
    postprocessed_data: list[dict] = [entry for entry in data.values()]

    # 5 --------------- Ingest data into Vector DB
    logger.info("Ingesting data into lancedb")
    db_manager.ingest_from_dicts(data=postprocessed_data)

    # 6 --------------- Encode search index
    index_size: int = len(indices)

    for i, (id, entry) in enumerate(indices.items()):
        image = load_image(entry["path"])
        inputs = engine.preprocess_image(image)

        logger.info(f'Encoding index of name "{id}" ({i + 1} of {index_size})...')

        outputs = engine.encode(inputs=inputs)

        vector = embedding_to_normalized_list(outputs)

        indices[id]["vector"] = vector

        # 7 --------------- Search and grab matches

        matches = db_manager.vector_search(vector)
        indices[id]["matches"] = sorted(matches, key=lambda x: x["_distance"])

    # 8 --------------- Classify correct matches
    results: list[tuple[int, bool]] = []
    successes: int = 0

    for id, index in indices.items():
        matches = index["matches"]
        matched_names = [int(m["name"]) for m in matches]

        found_match: bool = int(id) in matched_names

        if found_match:
            indices[id]["correct_match"] = True
            successes += 1

        results.append((int(id), found_match))

    # 9 --------------- Output score
    for result in results:
        print(f"INDEX {result[0]}: MATCH {'SUCCEEDED' if result[1] else 'FAILED'}")

    print(
        f"\n--- SUMMARY:\n- {successes} SUCCESSES\n- {len(indices) - successes} FAILED\n------------"
    )
