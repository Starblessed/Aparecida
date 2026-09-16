from pathlib import Path

from src.engine import Engine
from src.loader import load_image
from src.postprocessing import embedding_to_normalized_list


def main():
    MODEL_NAME: str = "google/siglip-so400m-patch14-384"
    IMAGE_PATH: Path = (
        Path(".").parent.absolute() / ".dev" / "little-cat-sitting-grass_1150-17019.png"
    )

    assert IMAGE_PATH.exists(), "Couldn't find the example image"

    try:
        # Loads model
        print("Initializing Model...")
        engine = Engine(model_name=MODEL_NAME)
        engine.initialize()

        print("Model initialized!")

        # Loads image
        print("Loading image...")
        image = load_image(IMAGE_PATH)

        print("Image loaded!")

        # Preprocesses and processes image
        print("Preprocessing image...")
        inputs = engine.preprocess_image(image)

        print("Image preprocessed!")

        print("Performing inference...")
        outputs = engine.inference(inputs=inputs)

        print("Inference done!")

        # Converts output to list
        print("Converting output to list")
        vector: list = embedding_to_normalized_list(outputs)

        print("Conversion done!")

        print("Pipeline finished with success!")
        print(f"Vector dims: {len(vector)}")

    except Exception as e:  # noqa: BLE001
        print(f"Error while running pipeline: {e!r}")


if __name__ == "__main__":
    main()
