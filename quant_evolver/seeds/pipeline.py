from __future__ import annotations

from .models import SeedCandidate, SeedEvaluation
from .scorer import SeedScorer
from .validator import SeedValidator
from .evaluator import SeedEvaluator


class SeedPipeline:
    def __init__(self, validator: SeedValidator | None = None, scorer: SeedScorer | None = None, evaluator: SeedEvaluator | None = None):
        self.validator = validator or SeedValidator()
        self.scorer = scorer or SeedScorer()
        self.evaluator = evaluator

    def run_static(self, seeds: list[SeedCandidate]) -> list[SeedEvaluation]:
        return [self.scorer.score(self.validator.validate(seed)) for seed in seeds]

    def run(self, seeds: list[SeedCandidate]) -> list[SeedEvaluation]:
        out: list[SeedEvaluation] = []
        for seed in seeds:
            ev = self.validator.validate(seed)
            if self.evaluator is not None and ev.valid:
                ev = self.evaluator.evaluate(ev)
            ev = self.scorer.score(ev)
            out.append(ev)
        return out
