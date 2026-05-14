from __future__ import annotations

from contextlib import contextmanager

from verl.utils.dataset.rl_dataset import RLHFDataset


@contextmanager
def _force_no_think(obj):
    if obj is None or not hasattr(obj, "apply_chat_template"):
        yield
        return
    original = obj.apply_chat_template

    def wrapped(*args, **kwargs):
        # Some reasoning-oriented chat templates accept enable_thinking=False.
        # Set it when supported while remaining harmless for tokenizers that
        # ignore unknown template kwargs.
        kwargs.setdefault("enable_thinking", False)
        return original(*args, **kwargs)

    obj.apply_chat_template = wrapped
    try:
        yield
    finally:
        obj.apply_chat_template = original


class NoThinkRLHFDataset(RLHFDataset):
    """RLHFDataset variant that disables reasoning-mode chat templates when supported."""

    def __getitem__(self, item):
        with _force_no_think(self.tokenizer), _force_no_think(self.processor):
            return super().__getitem__(item)
