# Certified decisions (`vybir.certify`)

Code: [`vybir/certify.py`](../vybir/certify.py) · experiments:
[`scripts/research_certify.py`](../scripts/research_certify.py) · raw results:
[`research/results.json`](../research/results.json) · full tables:
[`research/summary.md`](../research/summary.md) · earlier iterations:
[`research/history/`](../research/history/)

## The problem

A model that decides for people (a scam filter, a guard in front of a chatbot, ticket
routing) does not know when it is wrong. "Confidence 0.99" is a number, not a promise: models
can be confidently wrong. So critical decisions are either not automated at all, or automated
blindly.

## What vybir adds

Three layers on top of the local Laya decision model:

1. **Zero-shot.** Laya answers a typed question with no training at all.
2. **Residual marker adapter (few-shot).** Every option in a Laya prompt has its own `[MASK]`
   marker; its hidden state (1024 numbers) is what Laya's scorer reads. The adapter adds
   `w · marker_state + b_option` to Laya's calibrated logit for each option.
   - `w` is **shared by all options**, so the adapter has D + K parameters (about 1,026)
     whatever the number of options.
   - With `w = b = 0` it is **exactly** zero-shot Laya. An L2 penalty pulls it towards Laya,
     and its strength (including "bias only") is chosen by cross-validation on the labelled
     examples alone.
   - Trained with L-BFGS in PyTorch in about a second on a CPU.
3. **Conformal guarantee.** On separate labelled examples (calibration), a threshold `q̂`
   is computed; for a new input the prediction set is every option with `1 − p ≤ q̂`.
   Theorem (split conformal prediction): if new inputs come from the same distribution as the
   calibration examples, the true answer is inside the set with probability ≥ 1 − α.
   vybir acts automatically when the set holds **at most one** option and defers otherwise.
   A wrong automatic decision then always means the true label was outside the set, so

   **P(decided automatically AND wrong) ≤ α.**

   This holds for any model, even a bad one; a bad model simply decides less often.

Two stronger variants:

- **PAC (`delta=0.1`)**: the plain guarantee holds on average over calibration draws; the PAC
  threshold (exact binomial tail) makes it hold for *this* calibration set with probability
  ≥ 1 − δ.
- **Per class (`per_class=True`, Mondrian conformal)**: one threshold per true label, so the
  promise holds for every label separately. This is the one that matters for safety filters,
  see the results below.

```bash
vybir certify --data examples.jsonl --question question.json --alpha 0.05 --per-class --out cert.npz
vybir decide --cert cert.npz --state "URGENT! Your bank card has been blocked..."
```

## Protocol (fixed before the first result)

- Data (public, Hugging Face): `ucirvine/sms_spam` (5,574 SMS, 747 spam),
  `deepset/prompt-injections` (662, partly German), `jackhhao/jailbreak-classification`
  (1,306, balanced).
- The three Laya questions were written once and never tuned.
- 20 random splits: 40 % stratified test set, 200 calibration examples, and n labelled
  examples for the adapter drawn from the rest (at least 2 per class),
  n ∈ {8, 16, 32, 64, 128, 256, all}.
- Yardstick: TF-IDF (1–2-grams) + logistic regression with the same n labels. It is not part
  of vybir; it shows whether the method beats the simplest alternative.
- Laya features were extracted once on an RTX 3060 Laptop in float16: 7,542 texts in 44 s.

## Results (mean of 20 splits)

### Zero-shot and few-shot quality

| Task | Laya zero-shot (0 labels) acc / AUROC | vybir adapter, 64 labels | vybir adapter, all labels | TF-IDF, 64 labels | TF-IDF, all labels |
|---|---|---|---|---|---|
| SMS spam / scam | 0.766 / 0.932 | 0.910 / 0.934 | 0.947 / 0.963 | 0.866 / 0.950 | 0.951 / 0.992 |
| Prompt injection | 0.760 / 0.851 | 0.786 / 0.856 | 0.797 / 0.863 (197) | 0.671 / 0.819 | 0.760 / 0.906 (197) |
| Jailbreak | **0.988 / 1.000** | 0.992 / 0.999 | 0.994 / 1.000 | 0.841 / 0.980 | 0.957 / 0.990 |

### The guarantee, and why "per class" matters

α = 5 %. "Slip" = share of positives (spam, attacks) that were decided automatically as
harmless.

| Task, model | plain: auto / wrong&auto / **slip** | per class: auto / wrong&auto / **slip** |
|---|---|---|
| SMS, adapter 64 labels | 90.0 % / 4.78 % / **23.8 %** | 58.9 % / 4.82 % / **3.5 %** |
| SMS, Laya zero-shot | 51.8 % / 5.31 % / 1.2 % | 60.9 % / 4.56 % / 3.6 % |
| Prompt injection, Laya zero-shot | 48.8 % / 5.53 % / **14.0 %** | 42.6 % / 4.66 % / **5.05 %** |
| Prompt injection, adapter 64 | 50.8 % / 5.13 % / 9.2 % | 42.8 % / 4.32 % / 4.1 % |
| Jailbreak, Laya zero-shot | 100 % / 1.25 % / 0.2 % | 100 % / 1.25 % / 0.2 % |

An overall "≤ 5 % of all traffic wrong" is kept, yet at that promise a quarter of the scams
went through automatically. The per-class guarantee caps what hurts.

The plain guarantee is on average: across 20 splits the realised wrong-and-automatic rate at
α = 5 % ranged around 5 % (for example 5.31 % ± 0.31 % s.e. on SMS zero-shot). The PAC
variant stayed below: 2.7–3.6 % with 0–3 of 20 splits above α, as promised for δ = 0.1.

Fixed confidence thresholds are not a substitute: "act when p ≥ 0.99" decided nothing at all
with Laya zero-shot on SMS, and with an 8-label adapter it acted on half of the messages
while being wrong on 5.6 % of all of them.

## What did not work (reported as found)

- **v1:** with 8–32 labels the adapter *lowers* ranking quality (AUROC) below zero-shot Laya
  (SMS 0.896 vs 0.932, prompt injection 0.830 vs 0.851); its accuracy gains at small n come
  mostly from moving the decision threshold.
- **v2:** adding a "bias only" candidate to the cross-validation (designed after seeing v1)
  barely helped: SMS n = 8 AUROC 0.899, still below zero-shot.
- On SMS spam with 64+ labels, plain TF-IDF ranks better than the adapter (0.950–0.992 vs
  0.934–0.963): 2012-era SMS spam is lexical.
- **Distribution shift breaks the promise, as the theorem says it would.** A certifier
  calibrated on the 2012 SMS collection auto-approved a modern bank-phishing text
  ("your card has been blocked, confirm your PIN at …") with p = 0.99. The guarantee only
  covers inputs like the calibration data. With `--per-class` the same text was deferred.

## Iterations

| Version | Change | Why |
|---|---|---|
| v1 | adapter + split conformal, act on one-label sets | first protocol run |
| v2 | "bias only" candidate in the CV grid; PAC option | v1 showed ranking loss at small n; plain guarantee is only on average |
| v3 | act on sets with at most one label | an empty set also means "true label outside the set"; jailbreak went from 94.8 % to 100 % automatic at 1.25 % error |
| v4 | per-class (Mondrian) thresholds | a real-world test showed scams slipping through under the overall guarantee |

All versions were run on the same splits; v2–v4 were designed after seeing earlier results,
which is why every version's raw output is kept in `research/history/`.
