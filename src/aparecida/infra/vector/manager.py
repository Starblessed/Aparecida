from pathlib import Path

import lancedb
from pyarrow import Schema

from aparecida.infra.vector.data_loader import Loader


class VectorDatabaseManager:
    def __init__(self, uri: str):
        self.uri: str = uri
        self.db: lancedb.DBConnection = lancedb.connect(uri=uri)

    def create_table(self, name: str, schema: Schema):
        self.current_table = self.db.create_table(name=name, schema=schema)

    def open_table(self, name: str):
        self.current_table = self.db.open_table(name=name)

    def load_data_from_json(self, json_path: Path | str):
        loader: Loader = Loader()
        data: dict = loader.load_json(path=json_path)

        self.current_table.add(data=data)
