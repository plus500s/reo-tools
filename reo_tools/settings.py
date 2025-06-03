from dataclasses import dataclass

from reo_tools.cache import CSVCache, DataCache, NoOpCache
from reo_tools.reader import BaseReader, CSVReader


@dataclass(slots=True)
class PipelineSettings:
    reader: BaseReader = CSVReader()
    cache_backend: DataCache | NoOpCache = CSVCache()
    show_progress: bool = True
    recompute: bool = True
    coef_formula: str | None = None
    debug: bool = False
