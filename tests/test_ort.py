"""ORT export parity on a tiny model (skipped if onnx/onnxruntime are missing)."""

import numpy as np
import pytest
import torch

from vybir.export import DYNAMIC_AXES, INPUTS
from vybir.model import DecisionModelTorch, EncoderConfig

ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

TINY_CFG = {
    "model_type": "modernbert",
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "local_attention": 16,
    "max_position_embeddings": 256,
}


def test_ort_matches_torch_on_tiny(tmp_path):
    torch.manual_seed(0)
    model = DecisionModelTorch(
        EncoderConfig(TINY_CFG), {"head_layers": 1, "act_costs": {"escalate": 0.5}},
        attn_impl="math",
    ).eval()
    dummy = (
        torch.randint(1, 128, (2, 32)),
        torch.ones(2, 32, dtype=torch.long),
        torch.tensor([[5, 9], [6, 7]]),
        torch.ones(2, 2, dtype=torch.bool),
        torch.tensor([0, 2]),
    )
    out = tmp_path / "tiny.onnx"
    torch.onnx.export(
        model, dummy, str(out), input_names=INPUTS, output_names=["logits", "act"],
        dynamic_axes=DYNAMIC_AXES, opset_version=18, do_constant_folding=True,
    )
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = session.run(None, {name: value.numpy() for name, value in zip(INPUTS, dummy)})
    with torch.no_grad():
        expected = model(*dummy)
    for actual, reference in zip(got, expected):
        np.testing.assert_allclose(actual, reference.numpy(), atol=1e-4, rtol=1e-4)
