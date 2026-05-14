from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from .config import RFTConfig


@dataclass
class RFTLaunchResult:
    backend: str
    command: str
    log_path: Path | None = None
    pid_path: Path | None = None
    pid: int | None = None
    dry_run: bool = False
    notes: list[str] | None = None


class RFTBackend(ABC):
    def __init__(self, config: RFTConfig):
        self.config = config

    @abstractmethod
    def build_command(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def launch(self, dry_run: bool = False) -> RFTLaunchResult:
        raise NotImplementedError
