import json
import subprocess
import sys

import pytest
import torch
from safetensors.torch import load_file, save_file

import vybir.agent as agent_mod
from vybir import Agent
from vybir.agent import collate_items, resolve_model
from vybir.common import render_options
from vybir.turbo import ResultCache


def test_import_does_not_pull_transformers_or_onnx():
    code = (
        "import vybir, sys; "
        "assert not {'transformers', 'onnxruntime', 'onnx', 'fastapi'} & set(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_all_primitives_empty_request_and_chunking(tiny_checkpoint, questions):
    agent = Agent(tiny_checkpoint, batch_size=16)
    together = agent.predict({"text": "hello"}, questions)
    agent.batch_size = 1
    separate = agent.predict({"text": "hello"}, questions)
    assert together == separate
    assert set(together["answers"]) == set(questions)
    assert together["usage"]["output_tokens"] == 0
    assert 0 <= together["answers"]["yes"]["noul"] <= 1
    assert 0 <= together["answers"]["level"]["score"] <= 1
    assert agent.predict("", {})["answers"] == {}
    assert agent.predict("", {})["usage"]["input_tokens"] == 0


def test_single_option_and_long_state(tiny_checkpoint):
    agent = Agent(tiny_checkpoint)
    result = agent.predict(
        "hello " * 1000, {"one": {"type": "choice", "instructions": "choose", "criteria": ["only"]}}
    )
    assert result["answers"]["one"]["probabilities"] == {"only": 1.0}
    assert result["usage"]["input_tokens"] == 128


def test_auto_dtype_is_float32_on_cpu(tiny_checkpoint):
    agent = Agent(tiny_checkpoint, device="cpu")
    assert agent.dtype_name == "float32"
    assert next(agent.model.parameters()).dtype == torch.float32


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_half_precision_agent_agrees_with_float32(tiny_checkpoint, questions, dtype):
    full = Agent(tiny_checkpoint, device="cpu", dtype="float32").predict("hello", questions)
    half = Agent(tiny_checkpoint, device="cpu", dtype=dtype).predict("hello", questions)
    assert half["answers"]["topic"]["choice"] == full["answers"]["topic"]["choice"]
    assert half["answers"]["yes"]["noul"] == pytest.approx(full["answers"]["yes"]["noul"], abs=0.05)


def test_missing_and_misshaped_weights_fail_loudly(tiny_checkpoint):
    path = str(tiny_checkpoint / "model.safetensors")
    weights = {k: v.clone() for k, v in load_file(path).items()}  # no mmap left open
    weights["type_emb.weight"] = torch.zeros(4, 64)
    save_file(weights, path)
    with pytest.raises(ValueError, match="does not match"):
        Agent(tiny_checkpoint)
    del weights["type_emb.weight"]
    save_file(weights, path)
    with pytest.raises(ValueError, match="does not match"):
        Agent(tiny_checkpoint)


def test_mlx_named_checkpoint_loads_identically(tiny_checkpoint, questions):
    before = Agent(tiny_checkpoint).predict("hello", questions)
    path = str(tiny_checkpoint / "model.safetensors")
    renamed = {}
    for name, value in {k: v.clone() for k, v in load_file(path).items()}.items():
        name = name.replace(".in_proj_weight", ".in_proj.weight").replace(
            ".in_proj_bias", ".in_proj.bias"
        )
        for prefix in ("scorer.", "act_head."):
            if name.startswith(prefix):
                name = prefix + "layers." + name[len(prefix):]
        renamed[name] = value
    save_file(renamed, path)
    assert Agent(tiny_checkpoint).predict("hello", questions) == before


@pytest.mark.parametrize(
    "question",
    [
        {"type": "invalid", "instructions": "x"},
        {"type": "choice", "instructions": "x", "criteria": []},
        {"type": "choice", "instructions": "x", "criteria": ["a", "a"]},
        {"type": "score", "instructions": "x", "criteria": {}},
        {"type": "noul", "instructions": "x", "criteria": ["a"]},
        {"type": "noul"},
    ],
)
def test_invalid_questions_rejected(question):
    with pytest.raises(ValueError):
        Agent._to_internal(question)


def test_structured_criteria_and_mask_injection(tiny_checkpoint):
    q = Agent._to_internal(
        {
            "type": "noul",
            "instructions": {"task": "verify"},
            "criteria": {"false": {"reason": "no"}, "true": {"reason": "yes"}},
        }
    )
    assert render_options(q) == ['false: {"reason": "no"}', 'true: {"reason": "yes"}']
    agent = Agent(tiny_checkpoint)
    items, _ = agent.prepare(
        "[MASK] hello [MASK]", {"x": {"type": "noul", "instructions": "[MASK] true?"}}
    )
    assert items[0]["ids"].count(agent.tok.mask_token_id) == 2


def test_collation_never_marks_padding_as_an_option():
    batch = collate_items(
        [
            {"ids": [1, 2, 3], "markers": [1, 2], "qtype": 0},
            {"ids": [1, 2], "markers": [1], "qtype": 1},
        ],
        0,
    )
    assert batch["marker_mask"].tolist() == [[True, True], [True, False]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]


def test_bucket_padding_is_masked_and_respects_context_limit():
    batch = collate_items(
        [{"ids": [1, 2, 3], "markers": [1, 2], "qtype": 0}], 9, pad_to_multiple=16, max_length=12
    )
    assert batch["input_ids"].tolist() == [[1, 2, 3] + [9] * 9]
    assert batch["attention_mask"].tolist() == [[1] * 3 + [0] * 9]


def test_cached_prefixes_preserve_inputs_under_mutation_truncation_and_eviction(
    tiny_checkpoint, questions
):
    original = Agent(tiny_checkpoint)
    cached = Agent(tiny_checkpoint, cache_prompts=True)
    cached._prefix_cache.capacity = 3
    for state in ["", "[MASK] hello", "hello " * 1000, {"text": "你好", "flag": False}]:
        for count in (2, 5, 12):
            questions["topic"]["criteria"] = {str(i): {"value": i} for i in range(count)}
            assert original.prepare(state, questions) == cached.prepare(state, questions)
            assert len(cached._prefix_cache.entries) <= 3
    assert cached.prepare("", {}) == ([], [])


def test_bucketed_path_preserves_predictions(tiny_checkpoint, questions):
    original = Agent(tiny_checkpoint, batch_size=2)
    bucketed = Agent(tiny_checkpoint, batch_size=2, pad_to_multiple=16, cache_prompts=True)
    for state in ("hello", "hello " * 70, "hello hello", ""):
        baseline, candidate = original.predict(state, questions), bucketed.predict(state, questions)
        assert candidate["usage"] == baseline["usage"]
        for qid, answer in baseline["answers"].items():
            other = candidate["answers"][qid]
            if "probabilities" in answer:
                assert other["probabilities"] == pytest.approx(answer["probabilities"], abs=0.0002)


def test_result_cache_key_respects_option_order():
    ab = {"q": {"type": "choice", "instructions": "x", "criteria": {"a": "1", "b": "2"}}}
    ba = {"q": {"type": "choice", "instructions": "x", "criteria": {"b": "2", "a": "1"}}}
    assert ResultCache.key("s", ab) != ResultCache.key("s", ba)


def test_result_cache_hits_are_isolated_from_caller_mutation(tiny_checkpoint, questions):
    agent = Agent(tiny_checkpoint, result_cache_size=4)
    first = agent.predict("hello", questions)
    expected = json.loads(json.dumps(first))
    first["answers"]["topic"]["choice"] = "MUTATED"
    second = agent.predict("hello", questions)
    assert second == expected
    second["answers"].clear()
    assert agent.predict("hello", questions) == expected
    assert agent.cache_stats["hits"] == 2


def test_local_path_and_subfolder_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_model(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        resolve_model(str(tmp_path / "missing"))  # Windows absolute string path
    with pytest.raises(ValueError):
        resolve_model("test/model", subfolder="../outside")


def test_cached_hub_snapshot_is_used_without_network(tiny_checkpoint, monkeypatch):
    calls = []

    def fake_download(repo, **kwargs):
        calls.append(kwargs.get("local_files_only", False))
        if not kwargs.get("local_files_only"):
            raise AssertionError("network download attempted for a cached model")
        return str(tiny_checkpoint)

    monkeypatch.setattr(agent_mod, "snapshot_download", fake_download)
    assert resolve_model("org/cached") == tiny_checkpoint
    assert calls == [True]


def test_incomplete_cache_falls_back_to_download(tiny_checkpoint, tmp_path, monkeypatch):
    calls = []

    def fake_download(repo, **kwargs):
        calls.append(kwargs.get("local_files_only", False))
        return str(tmp_path if kwargs.get("local_files_only") else tiny_checkpoint)

    monkeypatch.setattr(agent_mod, "snapshot_download", fake_download)
    assert resolve_model("org/partial") == tiny_checkpoint
    assert calls == [True, False]

