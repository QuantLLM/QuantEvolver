from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path


def expr_hash(expr: str) -> str:
    return hashlib.sha1(" ".join(expr.strip().split()).encode("utf-8")).hexdigest()[:16]


class NoveltyArchive:
    """Persistent exact-expression archive used only for train reward shaping.

    Raw factor metrics remain untouched; this archive adds a search penalty so
    RL does not receive the same reward for repeatedly rediscovering an already
    known expression.
    """

    def __init__(self, path: str | Path | None, penalty: float = 0.25):
        self.path = Path(path) if path else None
        self.penalty = float(penalty)
        self._lock = threading.Lock()
        self.counts: dict[str, int] = {}
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    h = str(item.get("hash", ""))
                    c = int(item.get("count", 1))
                    if h:
                        self.counts[h] = max(self.counts.get(h, 0), c)
                except Exception:
                    continue
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def observe(self, expr: str, *, task_id: str = "") -> tuple[int, float, str]:
        h = expr_hash(expr)
        with self._lock:
            old = self.counts.get(h, 0)
            new = old + 1
            self.counts[h] = new
            penalty = self.penalty if old > 0 else 0.0
            if self.path:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"hash": h, "count": new, "expr": expr, "task_id": task_id}, ensure_ascii=False) + "\n")
        return old, penalty, h

    def seen_count(self, expr: str) -> tuple[int, str]:
        h = expr_hash(expr)
        with self._lock:
            return self.counts.get(h, 0), h

    def record(self, expr: str, *, task_id: str = "") -> tuple[int, str]:
        h = expr_hash(expr)
        with self._lock:
            new = self.counts.get(h, 0) + 1
            self.counts[h] = new
            if self.path:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"hash": h, "count": new, "expr": expr, "task_id": task_id}, ensure_ascii=False) + "\n")
        return new, h
