from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class LocalDataConfig:
    data_dir: str = "data"
    file_template: str = "{symbol_safe}_1m.parquet"
    time_column: str | None = None

    @classmethod
    def from_dict(cls, data: dict | None) -> "LocalDataConfig":
        data = data or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


class LocalDataStore:
    def __init__(self, config: LocalDataConfig | None = None):
        self.config = config or LocalDataConfig()

    def path_for(self, symbol: str) -> Path:
        return Path(self.config.data_dir) / self.config.file_template.format(symbol=symbol, symbol_safe=safe_symbol(symbol))

    def load(self, symbol: str, start, end) -> pd.DataFrame:
        path = self.path_for(symbol)
        if not path.exists():
            raise FileNotFoundError(f"Data for {symbol} not found: {path}")
        df = pd.read_parquet(path)
        if self.config.time_column and self.config.time_column in df.columns:
            df = df.set_index(pd.to_datetime(df[self.config.time_column]))
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        return df.loc[(df.index >= start_ts) & (df.index <= end_ts)].sort_index()

    def available_symbols(self) -> list[str]:
        root = Path(self.config.data_dir)
        template = self.config.file_template
        if "{symbol_safe}" in template:
            prefix, suffix = template.split("{symbol_safe}", 1)
            pattern = f"{prefix}*{suffix}"
            out = []
            for p in root.glob(pattern):
                name = p.name
                if not name.startswith(prefix) or not name.endswith(suffix):
                    continue
                safe = name[len(prefix): len(name) - len(suffix) if suffix else None]
                out.append(safe.replace("_", "/"))
            return sorted(out)
        return sorted(p.stem for p in root.glob("*.parquet"))
