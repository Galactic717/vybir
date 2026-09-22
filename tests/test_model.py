import numpy as np
import pytest
import torch

from vybir.model import (
    DecisionModelTorch,
    EncoderConfig,
    ModernBertTorch,
    attention_masks,
    remap_mlx_to_torch,
)

transformers = pytest.importorskip("transformers")


def hf_config(**kwargs):
    return transformers.ModernBertConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=3,
        num_attention_heads=1,
        local_attention=128,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        cls_token_id=2,
        sep_token_id=3,
        **kwargs,
    )


@pytest.mark.parametrize("impl", ["sdpa", "math"])
@pytest.mark.parametrize("local_theta", [10000.0, 160000.0])
def test_encoder_matches_transformers_across_window_and_padding(local_theta, impl):
    torch.manual_seed(0)
    cfg = hf_config(
        rope_parameters={
            "full_attention": {"rope_type": "default", "rope_theta": 160000.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": local_theta},
        }
    )
    reference = transformers.ModernBertModel(cfg).eval()
    model = ModernBertTorch(EncoderConfig(cfg.to_dict()), impl=impl).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    ids = torch.from_numpy(np.random.default_rng(1).integers(1, 128, size=(2, 145)))
    mask = torch.ones(2, 145, dtype=torch.long)
    mask[1, 13:] = 0  # padded queries beyond the local window must remain finite
    with torch.inference_mode():
        expected = reference(ids, attention_mask=mask).last_hidden_state
        actual = model(ids, mask)
    assert torch.isfinite(actual).all()
    valid = mask.bool()
    torch.testing.assert_close(actual[valid], expected[valid], atol=2e-5, rtol=2e-5)


def test_local_attention_includes_both_boundary_tokens():
    local = attention_masks(torch.ones(1, 140, dtype=torch.long), 128)["sliding_attention"][0, 0]
    assert local[70, 6] and local[70, 134]
    assert not local[70, 5] and not local[70, 135]


def test_unsupported_rope_scaling_is_rejected():
    cfg = hf_config().to_dict()
    cfg["rope_parameters"] = {"full_attention": {"rope_type": "linear", "factor": 2}}
    with pytest.raises(ValueError, match="unscaled"):
        EncoderConfig(cfg)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_forward_runs(dtype):
    # Regression: act_head got float32 features against half weights and crashed,
    # which broke dtype="float16" on CUDA and Router's default.
    torch.manual_seed(0)
    model = DecisionModelTorch(EncoderConfig(hf_config().to_dict()), {"head_layers": 1}).eval()
    model.to(dtype)
    batch = (
        torch.randint(1, 128, (2, 20)),
        torch.ones(2, 20, dtype=torch.long),
        torch.tensor([[3, 5], [4, 6]]),
        torch.ones(2, 2, dtype=torch.bool),
        torch.tensor([0, 2]),
    )
    with torch.inference_mode():
        logits, act = model(*batch)
    assert logits.dtype == act.dtype == torch.float32
    assert torch.isfinite(logits).all() and torch.isfinite(act).all()


def test_remap_mlx_names():
    out = remap_mlx_to_torch(
        {
            "head.layers.0.self_attn.in_proj.weight": torch.zeros(2, 2),
            "scorer.layers.1.weight": torch.zeros(2, 2),
            "act_head.layers.0.bias": torch.zeros(2),
        }
    )
    assert set(out) == {
        "head.layers.0.self_attn.in_proj_weight", "scorer.1.weight", "act_head.0.bias"
    }
