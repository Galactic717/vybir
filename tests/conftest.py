import json

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers

from vybir.model import DecisionModelTorch, EncoderConfig

TINY_ENCODER = {
    "model_type": "modernbert",
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_hidden_layers": 3,
    "num_attention_heads": 1,
    "local_attention": 16,
    "max_position_embeddings": 256,
}
TINY_AGENT = {
    "encoder": "test/tiny",
    "head_layers": 1,
    "max_len": 128,
    "head_max_len": 32,
    "act_costs": {"escalate": 0.5},
    "temperature": [1.3, 1.1, 2.0],
    "temperature_by_options": {"choice:2": 1.7},
}


@pytest.fixture
def tiny_checkpoint(tmp_path):
    path = tmp_path / "checkpoint"
    (path / "encoder").mkdir(parents=True)
    (path / "tokenizer").mkdir()
    (path / "encoder/config.json").write_text(json.dumps(TINY_ENCODER))
    (path / "rl_agent_config.json").write_text(json.dumps(TINY_AGENT))
    vocab = {t: i for i, t in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello"])}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(path / "tokenizer/tokenizer.json"))
    (path / "tokenizer/tokenizer_config.json").write_text(
        json.dumps(
            {"pad_token": "[PAD]", "cls_token": "[CLS]", "sep_token": "[SEP]", "mask_token": "[MASK]"}
        )
    )
    torch.manual_seed(7)
    model = DecisionModelTorch(EncoderConfig(TINY_ENCODER), TINY_AGENT)
    save_file(model.state_dict(), str(path / "model.safetensors"))
    return path


@pytest.fixture
def questions():
    return {
        "topic": {"type": "choice", "instructions": "Choose", "criteria": ["a", "b", "c"]},
        "level": {"type": "score", "instructions": "Level", "criteria": ["low", "high"]},
        "yes": {"type": "noul", "instructions": "Is this true?"},
    }
