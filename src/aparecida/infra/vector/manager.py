from pathlib import Path

import lancedb
from pyarrow import Schema

from aparecida.infra.vector.data_loader import Loader


class VectorDatabaseManager:
    def __init__(self, uri: str):
        self.uri: str = uri
        self.db: lancedb.DBConnection = lancedb.connect(uri=uri)

    def create_table(
        self,
        name: str,
        schema: Schema,
        overwrite_existing: bool = False,
        exist_ok: bool = False,
    ):
        self.current_table = self.db.create_table(
            name=name,
            schema=schema,
            mode="create" if not overwrite_existing else "overwrite",
            exist_ok=exist_ok,
        )

    def open_table(self, name: str):
        self.current_table = self.db.open_table(name=name)

    def start_table(
        self, name: str, overwrite_existing: bool = False, schema: Schema | None = None
    ):
        self.create_table(
            name=name,
            schema=schema,
            overwrite_existing=overwrite_existing,
            exist_ok=True,
        )

    def ingest_from_json(self, json_path: Path | str):
        loader: Loader = Loader()
        json_data: dict = loader.load_json(path=json_path)
        data: list[dict] = [value for value in json_data.values()]

        self.ingest_from_dicts(data)

    def ingest_from_dicts(self, data: list[dict]):
        self.current_table.add(data=data)

    def vector_search(self, vector: list[float], max_matches: int = 5):
        return (
            self.current_table.search(vector)
            .limit(max_matches)
            .select(["name", "id"])
            .to_list()
        )
