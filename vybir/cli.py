"""vybir: typed decisions (choice / score / noul) with Laya weights on any PC."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from . import __version__

DEFAULT_MODEL = "convaiinnovations/laya"
# Same request as examples/, inlined so `vybir bench` also works from a wheel install.
BENCH_STATE = {
    "from": "user@example.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "We were billed twice for March. Please refund the duplicate today "
    "or we will cancel our plan.",
}
BENCH_QUESTION = {
    "type": "choice",
    "instructions": "Which department should handle this email?",
    "criteria": {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages, system errors",
        "sales": "pricing, new contracts",
        "other": "everything else",
    },
}


def _agent_args(sub):
    sub.add_argument("--model", default=DEFAULT_MODEL, help="Hub id or local checkpoint dir")
    sub.add_argument("--subfolder", help="e.g. multilingual or typed-decisions")
    sub.add_argument("--revision")
    sub.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    sub.add_argument("--device", choices=("auto", "cuda", "gpu", "cpu"), default="auto")
    sub.add_argument("--backend", choices=("torch", "ort"), default="torch")
    sub.add_argument("--ort-path", help="ONNX file from `vybir export-onnx` (backend=ort)")
    sub.add_argument("--quantize", choices=("int8",), help="CPU weight-only int8 (torchao)")
    sub.add_argument("--compile", action="store_true", help="torch.compile the model")
    sub.add_argument("--threads", type=int, help="CPU threads for torch/ORT")
    sub.add_argument("--batch-size", type=int, default=16)


def _load(args):
    import torch

    from .agent import Agent

    if args.threads:
        torch.set_num_threads(args.threads)
    return Agent(
        args.model, device=args.device, dtype=args.dtype, revision=args.revision,
        subfolder=args.subfolder, batch_size=args.batch_size,
        compile=args.compile, quantize=args.quantize, backend=args.backend,
        ort_path=args.ort_path,
    )


def _read_questions(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _state(args):
    if args.state is not None:
        return args.state
    return json.loads(Path(args.state_file).read_text(encoding="utf-8"))


def cmd_predict(args):
    agent = _load(args)
    result = agent.predict(_state(args), _read_questions(args.questions))
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_route(args):
    from .router import Router

    questions = _read_questions(args.questions) if args.questions else None
    decision = Router().route(_state(args), questions, model=args.to, task=args.task, lang=args.lang)
    print(json.dumps(dict(decision), ensure_ascii=False, indent=2))


def cmd_bench(args):
    agent = _load(args)
    state, question = BENCH_STATE, BENCH_QUESTION
    one = {"department": question}
    for _ in range(3):
        agent.predict(state, one)
    samples = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        agent.predict(state, one)
        samples.append((time.perf_counter() - started) * 1000)
    samples.sort()
    many = {f"q{i}": question for i in range(args.questions)}
    started = time.perf_counter()
    agent.predict(state, many)
    seconds = time.perf_counter() - started
    import torch

    report = {
        "model": args.model, "backend": agent.backend, "device": str(agent.device),
        "dtype": agent.dtype_name, "threads": torch.get_num_threads(),
        "one_question_ms": {
            "p50": round(statistics.median(samples), 2),
            "p95": round(samples[max(0, int(len(samples) * 0.95) - 1)], 2),
            "runs": len(samples),
        },
        "batch": {"questions": args.questions, "seconds": round(seconds, 3),
                  "questions_per_second": round(args.questions / seconds, 1)},
    }
    print(json.dumps(report, indent=2))


def cmd_export(args):
    from .export import export_onnx

    path = export_onnx(args.model, args.output, subfolder=args.subfolder, revision=args.revision)
    print(f"exported {path}")


def cmd_certify(args):
    import numpy as np

    from .certify import Certifier, extract, take

    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    question = json.loads(Path(args.question).read_text(encoding="utf-8"))
    feats = extract(_load(args), [row["text"] for row in rows], question)
    index = {label: i for i, label in enumerate(feats["labels"])}
    unknown = {str(row["label"]) for row in rows} - set(index)
    if unknown:
        sys.exit(f"labels {sorted(unknown)} are not options of the question {sorted(index)}")
    y = np.array([index[str(row["label"])] for row in rows])
    order = np.random.default_rng(args.seed).permutation(len(y))
    n_test = int(len(y) * args.holdout)
    test, rest = order[:n_test], order[n_test:]
    cal, train = rest[: len(rest) // 2], rest[len(rest) // 2:]
    cert = Certifier()
    if len(np.unique(y[train])) > 1:
        cert.fit(take(feats, train), y[train], seed=args.seed)
    cert.calibrate(take(feats, cal), y[cal], alpha=args.alpha, per_class=args.per_class)
    cert.question = question
    cert.save(args.out)
    report = {"saved": str(args.out), "train": len(train), "calibration": len(cal),
              "adapter": cert.params is not None, "alpha": args.alpha,
              "per_class": args.per_class}
    if n_test:
        out = cert.predict(take(feats, test))
        wrong = out["label"] != y[test]
        report["held_out_check"] = {
            "examples": n_test,
            "accuracy": round(float(1 - wrong.mean()), 4),
            "coverage": round(float(out["sets"][np.arange(n_test), y[test]].mean()), 4),
            "decided_automatically": round(float(out["auto"].mean()), 4),
            "wrong_and_automatic": round(float((out["auto"] & wrong).mean()), 4),
            "wrong_and_automatic_by_true_label": {
                label: round(float((out["auto"] & wrong)[y[test] == k].mean()), 4)
                for k, label in enumerate(feats["labels"]) if (y[test] == k).any()
            },
        }
    print(json.dumps(report, indent=2))


def cmd_decide(args):
    import numpy as np

    from .certify import Certifier, extract

    cert = Certifier.load(args.cert)
    out = cert.predict(extract(_load(args), [_state(args)], cert.question))
    p = out["probabilities"][0]
    result = {
        "label": cert.labels[int(out["label"][0])],
        "probabilities": {label: round(float(v), 4) for label, v in zip(cert.labels, p)},
    }
    if "sets" in out:
        result["prediction_set"] = [lab for lab, keep in zip(cert.labels, out["sets"][0]) if keep]
        result["automatic"] = bool(out["auto"][0])
        per_label = np.ndim(cert.qhat) > 0
        result["guarantee"] = (
            f"for every true label: P(automatic and wrong) <= {cert.alpha}" if per_label
            else f"P(automatic and wrong) <= {cert.alpha}"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_serve(args):
    import os

    try:
        import uvicorn
    except ImportError:
        sys.exit("pip install 'vybir[server]' to use the HTTP server")
    os.environ.setdefault("VYBIR_MODEL", args.model)
    uvicorn.run("vybir.server:app", host=args.host, port=args.port)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):  # cp1251/cp437 consoles: never crash on emoji
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "snake":  # own parser: the live demo has many flags
        try:
            from .snake.cli import main as snake
        except ImportError:
            sys.exit("The Snake demo needs: pip install 'vybir[demo]'")

        return snake(argv[1:])
    if argv and argv[0] == "arcade":
        from .arcade.learn import main as arcade

        return arcade(argv[1:])
    parser = argparse.ArgumentParser(prog="vybir", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    sub = commands.add_parser("predict", help="Answer typed questions about a state")
    _agent_args(sub)
    source = sub.add_mutually_exclusive_group(required=True)
    source.add_argument("--state", help="Plain text input")
    source.add_argument("--state-file", type=Path, help="JSON state file")
    sub.add_argument("--questions", required=True, type=Path, help="JSON question definitions")
    sub.set_defaults(run=cmd_predict)

    sub = commands.add_parser("route", help="Show which checkpoint a request goes to (no load)")
    source = sub.add_mutually_exclusive_group(required=True)
    source.add_argument("--state")
    source.add_argument("--state-file", type=Path)
    sub.add_argument("--questions", type=Path)
    sub.add_argument("--to", help="Force english | multilingual | typed-decisions")
    sub.add_argument("--task")
    sub.add_argument("--lang")
    sub.set_defaults(run=cmd_route)

    sub = commands.add_parser("bench", help="Latency and throughput on this machine")
    _agent_args(sub)
    sub.add_argument("--iterations", type=int, default=20)
    sub.add_argument("--questions", type=int, default=50)
    sub.set_defaults(run=cmd_bench)

    sub = commands.add_parser("export-onnx", help="Export a checkpoint for --backend ort")
    sub.add_argument("--model", default=DEFAULT_MODEL)
    sub.add_argument("--subfolder")
    sub.add_argument("--revision")
    sub.add_argument("--output", type=Path, default=Path("models/vybir.onnx"))
    sub.set_defaults(run=cmd_export)

    sub = commands.add_parser(
        "certify", help="Adapt to your labelled examples and calibrate an error guarantee"
    )
    _agent_args(sub)
    sub.add_argument("--data", required=True, type=Path, help='JSONL: {"text": ..., "label": ...}')
    sub.add_argument("--question", required=True, type=Path, help="One question definition (JSON)")
    sub.add_argument("--alpha", type=float, default=0.05, help="Allowed P(automatic and wrong)")
    sub.add_argument("--holdout", type=float, default=0.2, help="Fraction kept to check the result")
    sub.add_argument("--per-class", action="store_true",
                     help="Guarantee alpha for every true label (e.g. scams), not just overall")
    sub.add_argument("--seed", type=int, default=0)
    sub.add_argument("--out", type=Path, default=Path("certifier.npz"))
    sub.set_defaults(run=cmd_certify)

    sub = commands.add_parser("decide", help="Certified decision, or defer to a human")
    _agent_args(sub)
    sub.add_argument("--cert", required=True, type=Path)
    source = sub.add_mutually_exclusive_group(required=True)
    source.add_argument("--state")
    source.add_argument("--state-file", type=Path)
    sub.set_defaults(run=cmd_decide)

    sub = commands.add_parser("serve", help="HTTP API: /predict /route /health")
    sub.add_argument("--model", default=DEFAULT_MODEL)
    sub.add_argument("--host", default="127.0.0.1")
    sub.add_argument("--port", type=int, default=8000)
    sub.set_defaults(run=cmd_serve)

    commands.add_parser("snake", help="Live Snake demo driven by real decisions (see --help)")
    commands.add_parser("arcade", help="Brains that learn Tetris, Snake, 2048 from the score")
    args = parser.parse_args(argv)
    return args.run(args)
