import numpy as np
import pytest

from vybir import Agent
from vybir.certify import Certifier, extract, take


def synthetic(n, k=2, d=8, informative=True, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, k)) * 2
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(k, p=row) for row in p]) if informative else rng.integers(0, k, n)
    markers = rng.normal(size=(n, k, d))
    markers[np.arange(n), y, 0] += 1.5  # the true option's state carries a signal Laya ignores
    feats = {"logits": logits.astype(np.float32), "markers": markers, "temperature": 1.0,
             "labels": [f"l{i}" for i in range(k)]}
    return feats, y


@pytest.mark.parametrize("informative", [True, False])
def test_conformal_coverage_holds_even_for_a_useless_model(informative):
    feats, y = synthetic(6000, informative=informative)
    cal, test = np.arange(1000), np.arange(1000, 6000)
    cert = Certifier().calibrate(take(feats, cal), y[cal], alpha=0.1)
    out = cert.predict(take(feats, test))
    covered = out["sets"][np.arange(len(test)), y[test]]
    assert covered.mean() >= 0.88  # guarantee is >= 0.90 in expectation
    wrong_auto = out["auto"] & (out["label"] != y[test])
    assert wrong_auto.mean() <= 0.12
    if not informative:
        assert out["auto"].mean() < 0.35  # a useless model mostly defers


def test_too_few_calibration_points_never_decide():
    feats, y = synthetic(20)
    out = Certifier().calibrate(feats, y, alpha=0.01).predict(feats)
    assert not out["auto"].any() and out["sets"].all()


def test_adapter_learns_signal_and_round_trips(tmp_path):
    feats, y = synthetic(600, informative=False, seed=1)
    train, cal, test = np.arange(200), np.arange(200, 400), np.arange(400, 600)
    zero = Certifier().predict(take(feats, test))["label"]
    cert = Certifier().fit(take(feats, train), y[train]).calibrate(take(feats, cal), y[cal], 0.1)
    out = cert.predict(take(feats, test))
    assert (out["label"] == y[test]).mean() > (zero == y[test]).mean() + 0.2
    cert.save(tmp_path / "c.npz")
    again = Certifier.load(tmp_path / "c.npz").predict(take(feats, test))
    np.testing.assert_allclose(again["probabilities"], out["probabilities"])
    assert (again["auto"] == out["auto"]).all()


def test_adapter_is_label_agnostic_across_option_counts():
    feats2, y2 = synthetic(400, k=2, informative=False, seed=2)
    feats4, y4 = synthetic(400, k=4, informative=False, seed=3)
    cert = Certifier().fit(feats2, y2)  # trained on 2-option questions ...
    w_only = Certifier()
    w_only.params = (*cert.params[:3], np.zeros(4))  # ... reused, bias dropped, on 4 options
    acc = (w_only.predict(feats4)["label"] == y4).mean()
    assert acc > (Certifier().predict(feats4)["label"] == y4).mean() + 0.2


def test_fit_rejects_single_class():
    feats, _ = synthetic(10)
    with pytest.raises(ValueError):
        Certifier().fit(feats, np.zeros(10, int))


def test_extract_matches_predict_and_keeps_order(tiny_checkpoint):
    agent = Agent(tiny_checkpoint, device="cpu", batch_size=2)
    question = {"type": "choice", "instructions": "Pick", "criteria": ["a", "b", "c"]}
    states = ["hello", "hello " * 30, "", "hello hello hello"]
    feats = extract(agent, states, question)
    assert feats["markers"].shape == (4, 3, 64)
    assert feats["labels"] == ["a", "b", "c"]
    probs = Certifier().probabilities(feats)
    for i, state in enumerate(states):
        expected = agent.predict(state, {"q": question})["answers"]["q"]["probabilities"]
        assert probs[i] == pytest.approx(list(expected.values()), abs=1e-4)
    single = extract(agent, [states[1]], question)
    np.testing.assert_allclose(single["markers"][0], feats["markers"][1], atol=1e-5)


def test_certify_and_decide_cli(tiny_checkpoint, tmp_path, capsys):
    import json

    from vybir.cli import main

    rows = [{"text": "hello " * (i % 7 + 1), "label": "b" if i % 3 else "a"} for i in range(60)]
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows))
    question = tmp_path / "q.json"
    question.write_text(json.dumps({"type": "choice", "instructions": "Pick", "criteria": ["a", "b"]}))
    cert = tmp_path / "cert.npz"
    main(["certify", "--model", str(tiny_checkpoint), "--data", str(data),
          "--question", str(question), "--alpha", "0.2", "--out", str(cert)])
    report = json.loads(capsys.readouterr().out)
    assert report["held_out_check"]["examples"] == 12 and report["adapter"]
    main(["decide", "--model", str(tiny_checkpoint), "--cert", str(cert), "--state", "hello"])
    decision = json.loads(capsys.readouterr().out)
    assert decision["label"] in ("a", "b") and isinstance(decision["automatic"], bool)
    assert set(decision["prediction_set"]) <= {"a", "b"}


def test_pac_calibration_holds_for_most_calibration_draws():
    feats, y = synthetic(40000, seed=5)
    test = np.arange(20000, 40000)
    below_plain = below_pac = 0
    draws = 150
    rng = np.random.default_rng(0)
    for _ in range(draws):
        cal = rng.choice(20000, 200, replace=False)
        for delta, name in ((None, "plain"), (0.1, "pac")):
            cert = Certifier().calibrate(take(feats, cal), y[cal], alpha=0.05, delta=delta)
            sets = cert.predict(take(feats, test))["sets"]
            covered = sets[np.arange(len(test)), y[test]].mean()
            if covered < 0.95:
                below_plain += name == "plain"
                below_pac += name == "pac"
    assert below_pac / draws <= 0.15  # promised <= delta = 0.10
    assert below_plain / draws > 0.3  # the plain guarantee is only on average


def test_per_class_guarantee_protects_the_rare_class(tmp_path):
    # 10 % positives, and a model that under-rates them: overall coverage looks fine while
    # positives slip through; per-class thresholds must bound the slip for each label.
    rng = np.random.default_rng(9)
    n = 40000
    y = (rng.random(n) < 0.1).astype(int)
    logits = np.column_stack([np.zeros(n), rng.normal(size=n) + 1.2 * y - 1.5])
    feats = {"logits": logits, "markers": np.zeros((n, 2, 1)), "temperature": 1.0,
             "labels": ["ok", "scam"]}
    cal, test = np.arange(2000), np.arange(2000, n)
    slip = {}
    for per_class in (False, True):
        cert = Certifier().calibrate(take(feats, cal), y[cal], alpha=0.05, per_class=per_class)
        out = cert.predict(take(feats, test))
        wrong = out["auto"] & (out["label"] != y[test])
        slip[per_class] = wrong[y[test] == 1].mean()
        assert wrong.mean() <= 0.06
    assert slip[False] > 0.2 and slip[True] <= 0.07
    cert.save(tmp_path / "pc.npz")
    again = Certifier.load(tmp_path / "pc.npz")
    assert np.allclose(again.qhat, cert.qhat)
