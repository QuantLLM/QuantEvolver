from __future__ import annotations

from .pure_verl_backend import PureVerlBackend
from .backend import RFTLaunchResult
from .config import RFTBackendName, RFTConfig


class RFTRunner:
    def __init__(self, config: RFTConfig):
        self.config = config

    def launch(self, dry_run: bool = False) -> RFTLaunchResult:
        if self.config.backend == RFTBackendName.PURE_VERL:
            return PureVerlBackend(self.config).launch(dry_run=dry_run)
        if self.config.backend == RFTBackendName.LOCAL_SMOKE:
            raise NotImplementedError(
                "local_smoke is for expression evaluation utilities, not long-running RFT launch."
            )
        raise ValueError(f"Unsupported RFT backend: {self.config.backend}")
