import json
from pathlib import Path
from typing import Annotated

import pyarrow as pa
from pydantic import BaseModel, Field

DIM: int = 1152

LANCE_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field("vector", pa.list_(pa.float32(), DIM), nullable=False),
        pa.field("source", pa.string(), nullable=False),
    ]
)


class PersonEntry(BaseModel):
    id: str
    name: str
    vector: Annotated[list[float], Field(min_length=DIM, max_length=DIM)]
    source: str

    @classmethod
    def from_dict(cls, obj: dict):
        return cls(
            id=obj["id"], name=obj["name"], vector=obj["vector"], source=obj["source"]
        )


class Loader:
    def __init__(self): ...

    def load_json(self, path: Path | str):
        with open(path, "r", encoding="utf8") as f:
            data = json.load(f)

        return data
