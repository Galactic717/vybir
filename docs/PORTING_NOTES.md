# Porting notes: from laya-mlx to vybir

How vybir was built, what was verified, and which defects were found on the way.
Measurements: Intel Core i7-12700H, NVIDIA RTX 3060 Laptop 6 GB, 16 GB RAM, Windows 11,
Python 3.14, torch 2.13 (CPU and cu126 builds), onnxruntime 1.24.

## 1. What Laya is

Laya (Convai Innovations, [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)) is not
a chat model. It answers typed questions about a state in one bidirectional forward pass:

```
[CLS] <type> question: <instructions> [SEP] [MASK] option0 [MASK] option1 ... [SEP] state [SEP]
        |
        v  ModernBERT encoder (bidirectional)
        +  type embedding (choice | score | noul)
        v  2 x TransformerEncoderLayer (pre-norm, ReLU)            <- decision head
        v  scorer: LayerNorm -> Linear -> GELU -> Linear -> one logit per [MASK]
        v  softmax with fitted temperatures -> choice | score | noul
        '- act_head(CLS + [top1, top1 - top2, entropy, k/255]) -> act_probability
```

| Part | Detail that matters for a faithful port |
|---|---|
| Encoder | ModernBERT-large: 28 layers x 1024, 16 heads. Every 3rd layer is global attention, the rest use a sliding window of 128 with an **inclusive** boundary (distance <= 64). NeoX-style RoPE with separate bases (160000 global, 10000 local). Layer 0 has no attention norm. GeGLU MLP `gelu(value) * gate`. |
| Decision head | Equivalent to `torch.nn.TransformerEncoderLayer(norm_first=True)`, which defaults to **ReLU** although the rest of the model uses GELU. |
| Prompt budget | `max_len` 512 / 1024; `head_max_len` 192 tokens for instructions plus all options; the state is truncated to fit. |
| Calibration | Temperatures per question type and option count, clamped to `[0.5, 5.0]`: the shipped `choice:11+` bucket is 0.1006 and would turn a coin flip into "99 % confident". Raw values stay available and a `RuntimeWarning` names clamped buckets. |
| Checkpoints | English 421M (512 tokens), multilingual 322M (mmBERT-base, 1024), typed-decisions 421M (1024). Weights ship as FP16. |

## 2. What laya-mlx contributed

[mizorewww/laya-mlx](https://github.com/mizorewww/laya-mlx) re-implemented the network in
Apple MLX and added a tidy runtime (batching, optional compilation, prefix reuse), a language
router, an embedding shortlist for large choice sets, presets, a Snake demo and a strong test
suite. It runs only on macOS arm64: `mlx` has no wheels elsewhere, and its live demo uses
POSIX-only `termios`.

## 3. The PyTorch port and what was wrong with the first version

vybir re-implements the network in PyTorch with the checkpoint's own parameter names, so
`load_state_dict(strict=True)` loads the published weights with no conversion. The first
version of the port had defects that were each reproduced with a script before fixing:

| # | Defect | Effect |
|---|---|---|
| 1 | `act_head` received float32 features while its weights were float16/bfloat16 | `RuntimeError: mat1 and mat2 must have the same dtype`. The CUDA path and the Router's default dtype were unusable. |
| 2 | Result-cache key built with `json.dumps(sort_keys=True)` | `{"a", "b"}` and `{"b", "a"}` share a key although option order is part of the prompt: a cache hit could return the answer to a different request. |
| 3 | The cache returned the stored object itself | A caller mutating the result (the HTTP server adds `latency_ms`) corrupted the cache. |
| 4 | Loading mismatched weights | The original error was swallowed and replaced by a confusing one; on Windows the memory map kept the file locked. |
| 5 | Model resolution | A Hub round-trip on every load; a Windows absolute path was treated as a repository id. |
| 6 | HTTP server | `model / task / lang` fields were accepted and ignored; two first requests loaded the model twice; bad questions returned 500. |
| 7 | ONNX export CLI | Crashed with `UnicodeEncodeError` on a cp1251 console (an emoji in the exporter's log). |

All are fixed and covered by regression tests.

## 4. How fidelity is verified

- Real weights: the encoder output equals `transformers.ModernBertModel` loaded with the same
  weights, **max abs difference 0.0** over 254 real tokens (`tests/test_integration.py`).
- Random tiny models: encoder matches Transformers within 2e-5 for both attention
  implementations, across the sliding-window boundary and padding (`tests/test_model.py`).
- ONNX export vs torch: 184/184 identical decisions over 40 real requests, max probability
  difference 0.0001 (`scripts/parity.py`).
- float16 on the RTX 3060 matches float32 on the CPU to about three decimals.
- The Snake engine replays laya-mlx's two published recordings board for board.

## 5. Speed work that was measured and rejected

- 6 threads on performance cores only: slower than 14 threads (258 ms vs 228 ms torch,
  196 ms vs 167 ms ONNX Runtime).
- Blockwise INT8 `MatMulNBits` in ONNX Runtime (block 32, int8 compute, AVX-VNNI): 207 ms,
  slower than float32, with probabilities drifting by up to 0.24. The CPU is compute-bound.
- torchao weight-only INT8: halves weight memory, no speed gain without a C++ compiler for
  `torch.compile`.

The real speedup is the GPU: 30 ms per question and 169 questions/s in float16 on the
RTX 3060, against 228 ms and 5.3 questions/s on the CPU.
