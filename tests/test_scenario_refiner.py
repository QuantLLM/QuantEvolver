from quant_evolver.scenarios.refiner import ScenarioRefiner


class FakeClient:
    def chat(self, messages, *, temperature=0.0, max_tokens=2048):
        return """
raw: test request
market: generic
universe: [ASSET]
frequency: 5min
horizon: 1
target: direction
factor_type: single_asset_timing
fields: [open, high, low, close, volume]
constraints: {}
preferred_signals: [momentum]
"""


def test_llm_scenario_refiner_parses_yaml():
    cfg = ScenarioRefiner(client=FakeClient()).refine("test request")
    assert cfg.universe == ["ASSET"]
    assert cfg.target == "direction"
    assert cfg.preferred_signals == ["momentum"]
