"""Real-checkpoint checks. Skipped unless convaiinnovations/laya is already cached."""

import json
from pathlib import Path

import pytest
import torch

from vybir import Agent

pytestmark = pytest.mark.integration
EXAMPLES = Path(__file__).parents[1] / "examples"


@pytest.fixture(scope="module")
def agent():
    from huggingface_hub import snapshot_download

    try:
        snapshot_download("convaiinnovations/laya", local_files_only=True)
    except Exception:
        pytest.skip("convaiinnovations/laya is not cached (hf download convaiinnovations/laya)")
    return Agent("convaiinnovations/laya", device="cpu")


def test_real_encoder_matches_transformers(agent):
    transformers = pytest.importorskip("transformers")
    cfg = transformers.ModernBertConfig(**agent.encoder_cfg)
    reference = transformers.ModernBertModel(cfg).eval()
    weights = {
        k.removeprefix("encoder."): v
        for k, v in agent.model.state_dict().items()
        if k.startswith("encoder.")
    }
    reference.load_state_dict(weights, strict=True)
    items, _ = agent.prepare(
        json.loads((EXAMPLES / "state.json").read_text(encoding="utf-8")),
        json.loads((EXAMPLES / "questions.json").read_text(encoding="utf-8")),
    )
    from vybir.agent import collate_items

    batch = collate_items(items, agent.tok.pad_token_id)
    with torch.inference_mode():
        expected = reference(batch["input_ids"], attention_mask=batch["attention_mask"])
        actual = agent.model.encoder(batch["input_ids"], batch["attention_mask"])
    valid = batch["attention_mask"].bool()
    err = (actual[valid] - expected.last_hidden_state[valid]).abs().max().item()
    assert err < 1e-4, err  # bit-identical on the dev machine; allow other-CPU rounding


def test_real_predictions_are_sensible(agent):
    out = agent.predict(
        json.loads((EXAMPLES / "state.json").read_text(encoding="utf-8")),
        json.loads((EXAMPLES / "questions.json").read_text(encoding="utf-8")),
    )
    answers = out["answers"]
    assert answers["department"]["choice"] == "billing"
    assert answers["refund"]["noul"] > 0.5
    assert out["usage"]["output_tokens"] == 0


def test_ort_matches_torch_when_exported(agent):
    path = Path(__file__).parents[1] / "models/vybir.onnx"
    if not path.exists():
        pytest.skip("run `vybir export-onnx` first")
    pytest.importorskip("onnxruntime")
    ort_agent = Agent("convaiinnovations/laya", device="cpu", backend="ort", ort_path=path)
    state = json.loads((EXAMPLES / "state.json").read_text(encoding="utf-8"))
    questions = json.loads((EXAMPLES / "questions.json").read_text(encoding="utf-8"))
    a, b = agent.predict(state, questions), ort_agent.predict(state, questions)
    for qid, answer in a["answers"].items():
        other = b["answers"][qid]
        assert other.get("choice") == answer.get("choice")
        for label, p in answer.get("probabilities", {}).items():
            assert other["probabilities"][label] == pytest.approx(p, abs=1e-3)
