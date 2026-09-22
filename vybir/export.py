"""Export a Laya checkpoint to ONNX for the ``backend='ort'`` runtime."""

from pathlib import Path

import torch

from .agent import Agent
from .model import DecisionModelTorch, EncoderConfig

INPUTS = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
DYNAMIC_AXES = {
    "input_ids": {0: "batch", 1: "seq"},
    "attention_mask": {0: "batch", 1: "seq"},
    "marker_pos": {0: "batch", 1: "markers"},
    "marker_mask": {0: "batch", 1: "markers"},
    "qtype": {0: "batch"},
    "logits": {0: "batch", 1: "markers"},
    "act": {0: "batch"},
}


def export_onnx(model_id_or_path, output, *, subfolder=None, revision=None, token=None):
    """Write a float32 ONNX graph with explicit (math) attention; never overwrites."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose a new path")
    try:
        import onnxscript  # noqa: F401  (torch.onnx's exporter needs it)
    except ImportError as error:
        raise RuntimeError("ONNX export needs: pip install 'vybir[ort]'") from error
    source = Agent(
        model_id_or_path, device="cpu", dtype="float32",
        subfolder=subfolder, revision=revision, token=token,
    )
    # ORT has no fused SDPA kernel for this pattern; the math variant is the same function.
    model = DecisionModelTorch(EncoderConfig(source.encoder_cfg), source.cfg, attn_impl="math")
    model.load_state_dict(source.model.state_dict(), strict=True)
    model.eval()
    dummy = (
        torch.randint(1, 1000, (2, 64)),
        torch.ones(2, 64, dtype=torch.long),
        torch.tensor([[5, 9, 0, 0], [6, 7, 0, 0]]),
        torch.tensor([[True, True, False, False], [True, True, False, False]]),
        torch.tensor([0, 2]),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(output),
        input_names=INPUTS, output_names=["logits", "act"],
        dynamic_axes=DYNAMIC_AXES, opset_version=18, do_constant_folding=True,
    )
    return output
