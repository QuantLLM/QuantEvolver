from __future__ import annotations

import shlex
import subprocess
from datetime import datetime
from pathlib import Path

from .backend import RFTBackend, RFTLaunchResult
from .config import RFTConfig
from quant_evolver.utils.config import save_config


class PureVerlBackend(RFTBackend):
    """QuantEvolver + Verl backend for GRPO/RFT training."""

    def build_command(self) -> str:
        c = self.config
        run_root = c.run_root_path
        conda_sh = str(Path(c.conda_sh).expanduser())
        cfg_path = run_root / f"{c.experiment_name}.pure_verl.yaml"
        save_config({"rft": c.to_dict()}, cfg_path)
        return (
            "set -eo pipefail\n"
            f"source {shlex.quote(conda_sh)}\n"
            f"conda activate {shlex.quote(c.train_conda_env)}\n"
            f"cd {shlex.quote(str(Path(__file__).resolve().parents[2]))}\n"
            f"export PYTHONPATH={shlex.quote(str(Path(__file__).resolve().parents[2]))}:$PYTHONPATH\n"
            "export NO_PROXY=127.0.0.1,localhost\n"
            "export no_proxy=127.0.0.1,localhost\n"
            f"python -m quant_evolver.rft.verl_main --config {shlex.quote(str(cfg_path))}\n"
        )

    def launch(self, dry_run: bool = False) -> RFTLaunchResult:
        c = self.config
        c.run_root_path.mkdir(parents=True, exist_ok=True)
        c.output_path.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_path = c.run_root_path / f"launch_{c.experiment_name}_pure_verl_{stamp}.sh"
        log_path = c.run_root_path / f"{c.experiment_name}_pure_verl_{stamp}.log"
        pid_path = c.run_root_path / f"{c.experiment_name}_pure_verl_{stamp}.pid"
        command = self.build_command()
        script_path.write_text(command, encoding="utf-8")
        script_path.chmod(0o755)
        if dry_run:
            return RFTLaunchResult(c.backend.value, command, log_path, pid_path, dry_run=True, notes=[f"wrote {script_path}"])
        log_f = open(log_path, "ab", buffering=0)
        proc = subprocess.Popen(["setsid", "bash", str(script_path)], stdout=log_f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        pid_path.write_text(str(proc.pid), encoding="utf-8")
        return RFTLaunchResult(c.backend.value, command, log_path, pid_path, pid=proc.pid, notes=[f"script={script_path}"])
