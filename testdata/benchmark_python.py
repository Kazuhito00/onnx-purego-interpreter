"""Benchmark all ONNX test models with onnxruntime and output JSON results."""

import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, numpy_helper

# Suppress ORT warnings (initializer-in-graph-inputs etc.)
ort.set_default_logger_severity(3)  # ERROR only

BASE = os.path.dirname(os.path.abspath(__file__))


def report_filter_tokens():
    raw = os.environ.get("REPORT_FILTER", "").strip()
    if not raw:
        return []
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def report_include_ops():
    value = os.environ.get("REPORT_INCLUDE_OPS", "").strip().lower()
    return value in {"1", "true", "yes", "on"}

def report_include_models():
    value = os.environ.get("ENABLE_MODEL_TESTS", "").strip().lower()
    if not value:
        return os.path.isdir(os.path.join(BASE, "models"))
    return value in {"1", "true", "yes", "on"}


def include_case(category, name):
    if category == "models" and not report_include_models():
        return False
    tokens = report_filter_tokens()
    if not tokens:
        return True
    if category == "ops" and not report_include_ops():
        return False
    target = f"{category}/{name}".lower()
    return any(token in target for token in tokens)


def load_tensor_proto(path):
    tp = TensorProto()
    with open(path, "rb") as f:
        tp.ParseFromString(f.read())
    return numpy_helper.to_array(tp), tp.name


def find_test_cases():
    """Find all test case directories under ops/ and models/."""
    cases = []
    for category in ["ops", "models"]:
        cat_dir = os.path.join(BASE, category)
        if not os.path.isdir(cat_dir):
            continue
        for name in sorted(os.listdir(cat_dir)):
            case_dir = os.path.join(cat_dir, name)
            model_path = os.path.join(case_dir, "model.onnx")
            if os.path.isfile(model_path):
                # Check for test_data_set_* (Model Zoo style)
                sub_dirs = sorted(
                    d
                    for d in os.listdir(case_dir)
                    if d.startswith("test_data_set_") and os.path.isdir(os.path.join(case_dir, d))
                )
                if sub_dirs:
                    for sd in sub_dirs:
                        case_name = f"{name}/{sd}"
                        if not include_case(category, case_name):
                            continue
                        cases.append(
                            {
                                "category": category,
                                "name": case_name,
                                "model_path": model_path,
                                "data_dir": os.path.join(case_dir, sd),
                            }
                        )
                else:
                    if not include_case(category, name):
                        continue
                    cases.append(
                        {
                            "category": category,
                            "name": name,
                            "model_path": model_path,
                            "data_dir": case_dir,
                        }
                    )
    return cases


def benchmark_case(case, n_runs=10):
    model_path = case["model_path"]
    data_dir = case["data_dir"]

    # Load inputs
    inputs = {}
    i = 0
    while True:
        path = os.path.join(data_dir, f"input_{i}.pb")
        if not os.path.exists(path):
            break
        arr, name = load_tensor_proto(path)
        inputs[name] = arr
        i += 1

    # Create session (measure load time once)
    t0 = time.perf_counter()
    sess = ort.InferenceSession(model_path)
    load_ms = (time.perf_counter() - t0) * 1000

    output_names = [o.name for o in sess.get_outputs()]

    # Warmup
    sess.run(output_names, inputs)

    # Benchmark
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(output_names, inputs)
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "name": case["name"],
        "category": case["category"],
        "load_ms": round(load_ms, 3),
        "run_avg_ms": round(np.mean(times), 3),
        "run_min_ms": round(np.min(times), 3),
        "run_max_ms": round(np.max(times), 3),
        "n_runs": n_runs,
    }


def main():
    cases = find_test_cases()
    print(f"Python benchmarks: {len(cases)} cases", file=sys.stderr)
    results = []
    for i, case in enumerate(cases):
        try:
            r = benchmark_case(case)
            r["status"] = "ok"
            results.append(r)
            print(f"  [{i+1}/{len(cases)}] {r['category']}/{r['name']}: avg={r['run_avg_ms']:.3f}ms", file=sys.stderr, flush=True)
        except Exception as e:
            results.append(
                {
                    "name": case["name"],
                    "category": case["category"],
                    "status": "error",
                    "error": str(e),
                }
            )
            print(f"  [{i+1}/{len(cases)}] {case['category']}/{case['name']}: ERROR {e}", file=sys.stderr, flush=True)

    # Output JSON to stdout
    json.dump(results, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
