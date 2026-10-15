#!/usr/bin/env python3
"""Run the pure-PP protocol for selected models and expose one verified wide table."""
import argparse
import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile

ROOT = Path("/data2/huangxin/qwen36_cache_experiments")
LLAMA_DIR = Path("/home/huangxin/code_list/BitNet/3rdparty/llama.cpp")
LIB_DIR = Path(__file__).resolve().parents[2] / "build_script/build_bin_qwen36_cuda/bin"
MANIFEST = ROOT / "bin/runtime.sha256"
EXPECTED_COMMIT = "373971d12e8bf094fac91667e42d732b5051f3bd"
MODES = ("recompute", "dram", "io-uring-direct")
METRICS = {
    "ttft_ms": "ms", "actual_prefill_eval_tps": "tokens/s", "effective_prompt_tps": "tokens/s",
    "conditioned_decode_forward_tps": "tokens/s", "generation_after_first_tps": "tokens/s",
    "end_to_end_output_tps": "tokens/s", "prompt_ready_ms": "ms", "last_sample_ms": "ms",
    "request_ms": "ms", "host_validation_ms": "ms", "parity_ms": "ms",
}
NA_METRICS = {"conditioned_decode_forward_tps", "generation_after_first_tps"}
NATIVE_FIELDS = ("mode", "metric", "unit", "n_expected", "n_observed", "status", "mean", "stddev")
WIDE_FIELDS = (
    "model", "test", "cache", "n", "ttft_mean_ms", "ttft_stddev_ms",
    "actual_prefill_mean_tps", "actual_prefill_stddev_tps",
    "effective_prefill_mean_tps", "effective_prefill_stddev_tps", "ctx", "run_dir",
)
TABLE_FORMAT = "{:<6} | {:<9} | {:<15} | {:>4} | {:>23} | {:>23} | {:>26}"
TABLE_HEADER = TABLE_FORMAT.format("model", "test", "cache", "n", "TTFT ms", "prefill eval t/s", "effective prompt t/s")


def decimal(text, low, high):
    if not re.fullmatch(r"0|[1-9][0-9]*", text) or len(text) > 10 or not low <= int(text) <= high:
        raise ValueError(f"Expected decimal integer in [{low}, {high}]: {text!r}")
    return int(text)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("probe", "pp-lengths"):
        parser.add_argument("--" + name, required=True)
    for name in ("bf16-model", "q4-model"):
        parser.add_argument("--" + name)
    for name, default in (("threads", "8"), ("gpu-id", "0"), ("repetitions", "3"), ("warmup", "1")):
        parser.add_argument("--" + name, default=default)
    args = parser.parse_args(argv)
    try:
        if args.bf16_model is None and args.q4_model is None:
            raise ValueError("Specify at least one of --bf16-model or --q4-model")
        args.threads = decimal(args.threads, 1, 2147483647)
        args.gpu_id = decimal(args.gpu_id, 0, 2147483647)
        args.repetitions = decimal(args.repetitions, 1, 1000)
        args.warmup = decimal(args.warmup, 1, 1000)
        args.pp_lengths = [decimal(value, 2, 262144) for value in args.pp_lengths.split(",")]
        if len(set(args.pp_lengths)) != len(args.pp_lengths):
            raise ValueError("PP lengths must be unique")
        for path in (args.probe, args.bf16_model, args.q4_model):
            if path is None:
                continue
            if not path:
                raise ValueError("Paths cannot be empty")
            if any(char in path for char in "\t\r\n"):
                raise ValueError("Paths cannot contain TSV control characters")
    except ValueError as error:
        parser.error(str(error))
    return args


def sha256(path):
    result = subprocess.run(["sha256sum", "--", str(path)], check=True, capture_output=True, text=True)
    fields = result.stdout.split()
    if not fields or not re.fullmatch(r"[0-9a-f]{64}", fields[0]):
        raise ValueError(f"Invalid SHA-256 for {path}")
    return fields[0]


def file_identity(path):
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size == 0:
        raise ValueError(f"Model missing, empty or not regular: {path}")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


@dataclass(frozen=True)
class Model:
    name: str
    path: Path
    digest: str
    identity: tuple


def check_runtime(runtime_id):
    head = subprocess.run(["git", "-C", str(LLAMA_DIR), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(LLAMA_DIR), "status", "--porcelain", "--untracked-files=no"],
                           check=True, capture_output=True, text=True).stdout.strip()
    if head != EXPECTED_COMMIT or dirty:
        raise ValueError("llama.cpp revision/tracked-source mismatch")
    if sha256(MANIFEST) != runtime_id:
        raise ValueError("Runtime manifest changed; refusing to mix builds")
    subprocess.run(["sha256sum", "--check", "--strict", "--quiet", str(MANIFEST)],
                   check=True, capture_output=True, text=True)


def check_frozen(runtime_id, models):
    check_runtime(runtime_id)
    for model in models:
        if file_identity(model.path) != model.identity:
            raise ValueError(f"Model changed during sweep: {model.path}")


def read_tsv(path, header):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", strict=True)
        if next(reader, None) != list(header):
            raise ValueError(f"Unexpected TSV header: {path}")
        rows = list(reader)
    if any(len(row) != len(header) for row in rows):
        raise ValueError(f"Malformed TSV row: {path}")
    return rows


def read_case(run_dir, model, pp, args, runtime_id):
    metadata = {}
    for key, value in read_tsv(run_dir / "metadata.tsv", ("key", "value")):
        if not key or key in metadata:
            raise ValueError(f"Duplicate/empty metadata key: {key}")
        metadata[key] = value
    required = {
        "benchmark_status": "complete", "model_parity": "passed_for_this_case_and_recorded_trials_only",
        "test": f"pp{pp}", "workload": "prefill-only", "prompt_tokens": str(pp),
        "sampled_outputs_per_request": "1", "decode_forwards_per_request": "0",
        "repetitions_per_mode": str(args.repetitions), "warmup_per_mode": str(args.warmup),
        "model": str(model.path), "model_sha256": model.digest, "runtime_id": runtime_id,
        "threads": str(args.threads), "cuda_visible_devices": str(args.gpu_id),
        "io_mode": "io_uring+O_DIRECT", "storage_backend": "ssd",
    }
    for key, value in required.items():
        if metadata.get(key) != value:
            raise ValueError(f"Metadata mismatch: {key}")
    actual_ctx = decimal(metadata.get("context_actual", ""), pp, 262144)
    metrics = {}
    for row in read_tsv(run_dir / "summary.tsv", NATIVE_FIELDS):
        mode, metric, unit, expected, observed, status, mean, stddev = row
        key = mode, metric
        if mode not in MODES or metric not in METRICS or key in metrics:
            raise ValueError(f"Unknown/duplicate summary key: {key}")
        if unit != METRICS[metric] or expected != str(args.repetitions) or observed != expected:
            raise ValueError(f"Summary unit/sample count mismatch: {key}")
        if metric in NA_METRICS:
            if (status, mean, stddev) != ("not_applicable", "NA", "NA"):
                raise ValueError(f"D=0 decode metric must be not_applicable/NA: {key}")
        else:
            numbers = float(mean), float(stddev)
            allow_zero = metric in {"host_validation_ms", "parity_ms"}
            if (status != "pass" or not all(math.isfinite(value) for value in numbers)
                    or numbers[0] < 0 or (numbers[0] == 0 and not allow_zero) or numbers[1] < 0
                    or (args.repetitions == 1 and numbers[1] != 0)):
                raise ValueError(f"Invalid summary status/value: {key}")
        metrics[key] = mean, stddev
    if set(metrics) != {(mode, metric) for mode in MODES for metric in METRICS}:
        raise ValueError("Missing summary mode/metric rows")
    rows = []
    for mode in MODES:
        values = [model.name, f"pp{pp}", mode, str(args.repetitions)]
        for metric in ("ttft_ms", "actual_prefill_eval_tps", "effective_prompt_tps"):
            values.extend(metrics[mode, metric])
        values.extend((str(actual_ctx), str(run_dir)))
        rows.append(dict(zip(WIDE_FIELDS, values)))
    return rows


def exclusive_text(path):
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
                     "w", encoding="utf-8", newline="")


def publish_summary(sweep, rows):
    fd, name = tempfile.mkstemp(prefix=".summary.", suffix=".tsv", dir=sweep)
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=WIDE_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(name, sweep / "summary.tsv")


def run(args):
    probe = ROOT / "bin/qwen36-prefix-snapshot"
    if Path(args.probe).resolve(strict=True) != probe.resolve(strict=True) or not os.access(probe, os.X_OK):
        raise ValueError("--probe must be the fingerprinted ROOT/bin/qwen36-prefix-snapshot")
    if not any(re.fullmatch(r"[0-9a-f]{64} [ *]" + re.escape(str(probe)), line)
               for line in MANIFEST.read_text(encoding="utf-8").splitlines()):
        raise ValueError("Fixed probe is absent from runtime manifest")
    inputs = []
    for name, path in (("BF16", args.bf16_model), ("Q4_0", args.q4_model)):
        if path is None:
            continue
        path = Path(path).resolve(strict=True)
        if any(char in str(path) for char in "\t\r\n"):
            raise ValueError("Canonical model paths cannot contain TSV control characters")
        inputs.append((name, path, file_identity(path)))
    print("Preparing runtime and selected model hashes outside request timings.", flush=True)
    runtime_id = sha256(MANIFEST)
    check_runtime(runtime_id)
    models = []
    for name, path, identity in inputs:
        digest = sha256(path)
        if file_identity(path) != identity:
            raise ValueError(f"Model changed while hashing: {path}")
        models.append(Model(name, path, digest, identity))
    check_frozen(runtime_id, models)
    runs = ROOT / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    sweep = Path(tempfile.mkdtemp(prefix="sweep.", dir=runs))
    for name in ("cases", "logs"):
        (sweep / name).mkdir(mode=0o700)
    with exclusive_text(sweep / "runtime-id") as handle:
        handle.write(runtime_id + "\n")
    with exclusive_text(sweep / "metadata.tsv") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerows([
            ("key", "value"),
            ("pp_lengths", ",".join(map(str, args.pp_lengths))),
            ("threads", str(args.threads)),
            ("cuda_visible_devices", str(args.gpu_id)),
            ("repetitions_per_mode", str(args.repetitions)),
            ("warmup_per_mode", str(args.warmup)),
            ("model_groups", ",".join(model.name for model in models)),
            ("runtime_id", runtime_id),
        ])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    env["LD_LIBRARY_PATH"] = str(LIB_DIR) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    rows = []
    with exclusive_text(sweep / "index.tsv") as index:
        writer = csv.writer(index, delimiter="\t", lineterminator="\n")
        writer.writerow(("status", "model", "pp", "ctx", "run_dir"))
        index.flush()
        print(f"Sweep: {sweep}\nNative logs: {sweep / 'logs'}", flush=True)
        print(TABLE_HEADER, flush=True)
        for model in models:
            for pp in args.pp_lengths:
                ctx = (pp + 255) // 256 * 256
                case = f"{model.name}-pp{pp}"
                run_dir, log_path = sweep / "cases" / case, sweep / "logs" / (case + ".log")
                event = (model.name, pp, ctx, str(run_dir))
                writer.writerow(("started", *event))
                index.flush()
                try:
                    run_dir.mkdir(mode=0o700)
                    with exclusive_text(log_path) as log:
                        check_frozen(runtime_id, models)
                        command = [str(probe), "--model", str(model.path), "--model-sha256", model.digest,
                                   "--runtime-id", runtime_id, "--output-dir", str(run_dir),
                                   "--threads", str(args.threads), "--prefix-tokens", str(pp - 1),
                                   "--suffix-tokens", "1", "--decode-tokens", "0", "--ctx-size", str(ctx),
                                   "--repetitions", str(args.repetitions), "--warmup", str(args.warmup),
                                   "--batch-size", "2048", "--ubatch-size", "512", "--seed", "1",
                                   "--max-blob-mib", "8192", "--max-store-mib", "16384", "--disk-reserve-mib", "16384"]
                        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=sweep)
                        check_frozen(runtime_id, models)
                        if completed.returncode:
                            raise ValueError(f"Probe exited with status {completed.returncode}")
                    added = read_case(run_dir, model, pp, args, runtime_id)
                    publish_summary(sweep, rows + added)
                    rows.extend(added)
                    writer.writerow(("pass", *event))
                    index.flush()
                except (OSError, ValueError, csv.Error, subprocess.SubprocessError) as error:
                    writer.writerow(("failed", *event))
                    index.flush()
                    raise RuntimeError(f"FAILED {model.name} pp{pp}; log: {log_path}; {error}") from error
                for row in added:
                    print(TABLE_FORMAT.format(row["model"], row["test"], row["cache"], row["n"],
                          f"{float(row['ttft_mean_ms']):.2f} +/- {float(row['ttft_stddev_ms']):.2f}",
                          f"{float(row['actual_prefill_mean_tps']):.2f} +/- {float(row['actual_prefill_stddev_tps']):.2f}",
                          f"{float(row['effective_prefill_mean_tps']):.2f} +/- {float(row['effective_prefill_stddev_tps']):.2f}"), flush=True)
    print(f"Summary: {sweep / 'summary.tsv'}\nIndex: {sweep / 'index.tsv'}\nLogs: {sweep / 'logs'}", flush=True)
    return sweep


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, csv.Error, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
