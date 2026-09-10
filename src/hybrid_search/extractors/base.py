from pathlib import Path

from ..ir import FunctionIR
from .java_extractor import JavaExtractor
from .python_extractor import PythonExtractor


class CodebaseIndexer:
    def __init__(self) -> None:
        self.extractors = {".py": PythonExtractor(), ".java": JavaExtractor()}

    def index_path(self, path: str | Path) -> list[FunctionIR]:
        root = Path(path)
        if root.is_file():
            return self.index_file(root)
        functions: list[FunctionIR] = []
        for file_path in root.rglob("*"):
            if file_path.is_file() and file_path.suffix in self.extractors:
                functions.extend(self.index_file(file_path))
        return functions

    def index_file(self, path: str | Path) -> list[FunctionIR]:
        file_path = Path(path)
        extractor = self.extractors.get(file_path.suffix)
        if extractor is None:
            return []
        source = file_path.read_text(encoding="utf-8")
        return extractor.extract(source, str(file_path))
