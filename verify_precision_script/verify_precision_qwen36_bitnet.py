#!/usr/bin/env python3
"""Compare existing BF16/Q4_0/I2_S models on local text and fixed scored smoke tests.
Requires numpy and the current child source's gguf-py. Never downloads or changes models.
Exit: 0 = tested-slice pass, 1 = execution/parse failure, 2 = quality not passed,
3 = unverified (insufficient data, checks, or separation of PPL error bars).
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
LLAMA = Path("/home/huangxin/code_list/BitNet/3rdparty/llama.cpp")
MODEL_DIR = Path("/data3/huangxin/model_list/Qwen3.6-35B-A3B")
MODELS = {"BF16": "Qwen3.6-35B-A3B-bf16.gguf", "Q4_0": "Qwen3.6-35B-A3B-Q4_0.gguf", "I2_S": "Qwen3.6-35B-A3B-1.58bit.gguf"}
TEMPLATE = r"""{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n<think>\n\n</think>\n\n' }}{% endif %}"""
SMOKE = [
    ("english", "What is the capital of France? Reply with only the city name.", "Paris"),
    ("chinese", "中国的首都是哪里？只回答城市名称。", "北京"),
    ("arithmetic", "Compute 17 + 25. Reply with only the integer.", "42"),
    ("python_output", "What is sum(x*x for x in range(4)) in Python? Reply with only the integer.", "14"),
    ("python_function", "Write only Python code defining add(a, b) that returns a + b.", "def add(a, b): return a + b"),
]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def tokenizer_hash(model):
    # Read-only mmap; hash metadata, not the large tensor payloads.
    from gguf import GGUFReader
    reader = GGUFReader(model)
    for key in ("tokenizer.ggml.model", "tokenizer.ggml.tokens"):
        if key not in reader.fields:
            raise ValueError(f"{model}: missing {key}")
    digest = hashlib.sha256()
    for key, field in sorted(reader.fields.items()):
        if key.startswith("tokenizer.") and not key.startswith("tokenizer.chat_template"):
            for part in field.parts:
                raw = part.tobytes()
                digest.update(len(raw).to_bytes(8, "little"))
                digest.update(raw)
    return digest.hexdigest()


def run(command, stem, metrics):
    command = [str(x) for x in command]
    print("$ " + shlex.join(command), flush=True)
    stem.with_suffix(".command.txt").write_text(shlex.join(command) + "\n")
    # Do not inherit LLAMA_ARG_* overrides for templates, sampling, or contexts.
    env = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_ARG_")}
    env["LC_ALL"] = "C"
    with stem.with_suffix(".stdout.log").open("wb") as out, stem.with_suffix(".stderr.log").open("wb") as err:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=out, stderr=err, env=env)
    metrics["commands"].append({"argv": command, "returncode": result.returncode, "log_stem": str(stem)})
    if result.returncode:
        raise RuntimeError(f"exit {result.returncode}: {stem}; see raw logs")
    return stem.with_suffix(".stdout.log").read_text(), stem.with_suffix(".stderr.log").read_text()


def parse_ppl(text, ctx, requested):
    if "you need at least" in text and "tokens to evaluate perplexity" in text:
        return {"status": "unverified", "reason": "corpus slice has fewer than 2 * context tokens"}
    header = re.search(r"calculating perplexity over (\d+) chunks, n_ctx=(\d+)", text)
    final = re.findall(r"Final estimate: PPL = (\S+) \+/- (\S+)", text)
    steps = re.findall(r"\[(\d+)\]([^,\s]+),", text)
    if not header or len(final) != 1:
        raise ValueError("missing/ambiguous final PPL estimate or evaluated chunk range")
    chunks = int(header[1])
    if int(header[2]) != ctx or not 0 < chunks <= requested or [int(i) for i, _ in steps] != list(range(1, chunks + 1)):
        raise ValueError("inconsistent PPL context/chunks or incomplete cumulative results")
    ppl, error = map(float, final[0])
    values = [float(v) for _, v in steps]
    if not all(math.isfinite(x) and x > 0 for x in [ppl] + values) or not math.isfinite(error) or error < 0:
        raise ValueError("non-finite/invalid PPL")
    if values[-1] != ppl:
        raise ValueError("final and cumulative PPL disagree")
    return {"status": "measured", "mean_ppl": ppl, "ppl_standard_error": error,
            "aggregation": "exp(mean token negative log-likelihood), not mean of cumulative PPLs",
            "chunks": chunks, "requested_chunks": requested, "cumulative_ppl": values,
            "context_token_range_half_open": [0, chunks * ctx],
            "scored_token_ranges_half_open": [[i * ctx + ctx // 2 + 1, (i + 1) * ctx] for i in range(chunks)],
            "scored_tokens": chunks * (ctx - ctx // 2 - 1)}


def response_text(stdout, prompt):
    # Current tools/cli/cli.cpp always echoes the prompt, even with --no-display-prompt.
    marker = "\n> " + prompt + "\n\n"
    if stdout.count(marker) != 1 or not stdout.endswith("\nExiting...\n"):
        raise ValueError("unexpected llama-cli output framing; cannot isolate the response")
    answer = stdout.split(marker, 1)[1].rsplit("\nExiting...\n", 1)[0].strip()
    if not answer or "[Start thinking]" in answer or "Error:" in answer:
        raise ValueError("empty response, unexpected reasoning, or CLI error")
    return answer


def score(name, answer, expected):
    if name == "python_function":
        code = re.sub(r"\A```(?:python)?\s*\n(.*?)\n```\Z", r"\1", answer, flags=re.S)
        try:
            return ast.dump(ast.parse(code)) == ast.dump(ast.parse(expected))
        except SyntaxError:
            return False
    return answer.strip().rstrip(".。!").casefold() == expected.casefold()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="local UTF-8 fixed corpus, normally WikiText-2 test wiki.test.raw")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--bin-dir", type=Path, default=ROOT / "build_script/build_bin_qwen36_bitnet/bin")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--max-chars", type=int, default=65536, help="evaluate the first N Unicode characters of --data (saved verbatim)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "verify_precision_script/results_qwen36_bitnet", help="parent of a new unique run directory")
    args = parser.parse_args()
    if min(args.threads, args.max_chars) < 1 or args.ctx_size < 128 or args.chunks < 2:
        parser.error("threads/max-chars must be positive, ctx-size >= 128, chunks >= 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-"), dir=args.output_dir.resolve()))
    print(f"Results: {out}", flush=True)
    metrics = {"quality": "unverified", "errors": [], "reasons": [], "models": {}, "commands": [],
               "settings": {"context": args.ctx_size, "threads": args.threads, "seed": 0, "temperature": 0,
                            "kv": "f16", "flash_attention": "auto", "gpu_layers": 0, "predict": 96},
               "scope": "Limited local corpus and fixed smoke checks only; no general quality or SWE preservation claim."}
    exit_code = 1
    try:
        raw = args.data.read_bytes()
        text = raw.decode("utf-8")[:args.max_chars]
        if not text.strip():
            raise ValueError("empty corpus slice")
        data = text.encode("utf-8")
        corpus = out / "corpus.slice.txt"
        corpus.write_bytes(data)
        template = out / "chat-template.jinja"
        template.write_text(TEMPLATE)
        metrics["data"] = {"source_path": str(args.data.resolve()), "source_sha256": sha256(raw),
                           "provenance": "user-supplied local text; WikiText-2 test provenance not independently verified",
                           "expected_public_source": "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip",
                           "submitted_character_range_half_open": [0, len(text)], "submitted_bytes": len(data),
                           "submitted_slice_sha256": sha256(data), "saved_slice": str(corpus),
                           "range_note": "Hash covers the submitted UTF-8 slice, not just scored tokens. Exact evaluated/scored token offsets are recorded per model; trailing text may be unused. Token offsets include tokenizer special-token policy."}
        metrics["template_sha256"] = sha256(TEMPLATE.encode())
        sys.path.insert(0, str(LLAMA / "gguf-py"))
        binaries = {name: args.bin_dir.resolve() / name for name in ("llama-perplexity", "llama-cli")}
        for name, binary in binaries.items():
            if not os.access(binary, os.X_OK):
                raise ValueError(f"missing executable: {binary}")
            metrics[name] = {"path": str(binary), "sha256": sha256(binary.read_bytes())}
            stdout, stderr = run([binary, "--help"], out / f"help-{name}", metrics)
            required = (["--chunks", "--ppl-stride", "--ppl-output-type"] if name == "llama-perplexity" else
                        ["--single-turn", "--simple-io", "--chat-template-file", "--reasoning", "--no-show-timings"])
            if any(flag not in stdout + stderr for flag in required):
                raise ValueError(f"{binary}: required flags missing from --help")
        for label, filename in MODELS.items():
            model = args.model_dir.resolve() / filename
            metrics["models"][label] = {"path": str(model), "bytes": model.stat().st_size,
                                        "tokenizer_sha256": tokenizer_hash(model), "smoke": {}}
        if len({m["tokenizer_sha256"] for m in metrics["models"].values()}) != 1:
            raise ValueError("model tokenizer metadata differs; comparison refused")
        common = ["-t", args.threads, "-tb", args.threads, "-c", args.ctx_size,
                  "-b", min(args.ctx_size, 512), "-ub", min(args.ctx_size, 512), "-ngl", 0,
                  "-ctk", "f16", "-ctv", "f16", "-fa", "auto", "--fit", "off", "--no-escape"]
        for label, model in metrics["models"].items():
            try:
                stdout, stderr = run([binaries["llama-perplexity"], "-m", model["path"], *common,
                                      "-f", corpus, "--chunks", args.chunks, "--ppl-stride", 0,
                                      "--ppl-output-type", 0], out / f"{label}-ppl", metrics)
                model["ppl"] = parse_ppl(stdout + "\n" + stderr, args.ctx_size, args.chunks)
                print(f"{label} PPL: {model['ppl']}", flush=True)
            except (OSError, ValueError, RuntimeError) as exc:
                metrics["errors"].append(f"{label} PPL: {exc}")
            for name, prompt, expected in SMOKE:
                try:
                    stdout, _ = run([binaries["llama-cli"], "-m", model["path"], *common,
                                     "--single-turn", "--simple-io", "--color", "off", "--no-show-timings",
                                     "--jinja", "--chat-template-file", template, "--reasoning", "off",
                                     "--reasoning-budget", 0, "--temp", 0, "--seed", 0, "--repeat-penalty", 1,
                                     "--system-prompt", "Follow the requested answer format exactly.",
                                     "-n", 96, "-p", prompt], out / f"{label}-{name}", metrics)
                    answer = response_text(stdout, prompt)
                    passed = score(name, answer, expected)
                    model["smoke"][name] = {"passed": passed, "prompt": prompt, "expected": expected, "answer": answer}
                    print(f"{label} {name}: {'PASS' if passed else 'FAIL'} {answer!r}", flush=True)
                except (OSError, ValueError, RuntimeError) as exc:
                    metrics["errors"].append(f"{label} {name}: {exc}")
        measured = {k: v["ppl"] for k, v in metrics["models"].items() if v.get("ppl", {}).get("status") == "measured"}
        for label, result in measured.items():
            result["changes"] = {base: {"absolute": result["mean_ppl"] - measured[base]["mean_ppl"],
                                       "percent": 100 * (result["mean_ppl"] / measured[base]["mean_ppl"] - 1)}
                                 for base in ("BF16", "Q4_0") if base in measured}
        comparable = len(measured) == 3 and len({p["chunks"] for p in measured.values()}) == 1
        baseline_passes = {name for base in ("BF16", "Q4_0") for name, r in metrics["models"][base]["smoke"].items() if r["passed"]}
        lost = sorted(name for name in baseline_passes if not metrics["models"]["I2_S"]["smoke"].get(name, {}).get("passed"))
        metrics["lost_baseline_checks"] = lost
        bf16_comparable = all(k in measured for k in ("BF16", "I2_S")) and measured["BF16"]["chunks"] == measured["I2_S"]["chunks"]
        increased = bf16_comparable and measured["I2_S"]["mean_ppl"] > measured["BF16"]["mean_ppl"]
        if increased or lost:
            metrics["quality"] = "not_passed"
            metrics["reasons"].append("I2_S PPL increased versus BF16 or baseline-passing smoke checks were lost; no tolerance applied")
        elif (not comparable or any(p["chunks"] < args.chunks for p in measured.values()) or
              not baseline_passes or metrics["errors"]):
            metrics["reasons"].append("insufficient/incomparable data, no baseline-passing checks, or execution/parse failure")
        elif measured["I2_S"]["mean_ppl"] + 2 * measured["I2_S"]["ppl_standard_error"] >= measured["BF16"]["mean_ppl"] - 2 * measured["BF16"]["ppl_standard_error"]:
            metrics["reasons"].append("no observed regression, but reported PPL error bars overlap; non-degradation remains unverified")
        else:
            metrics["quality"] = "passed_on_tested_slice_only"
            metrics["reasons"].append("no observed regression on this slice and no lost baseline checks; separated reported error bars are not a general statistical guarantee")
        exit_code = 1 if metrics["errors"] else {"not_passed": 2, "unverified": 3, "passed_on_tested_slice_only": 0}[metrics["quality"]]
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        metrics["errors"].append(str(exc))
    finally:
        metrics["exit_code"] = exit_code
        (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        lines = [f"Quality: {metrics['quality']}", *metrics["reasons"], *metrics["errors"]]
        for label, model in metrics["models"].items():
            ppl = model.get("ppl", {})
            lines.append(f"{label}: mean PPL={ppl.get('mean_ppl', 'unverified')}, changes={ppl.get('changes', {})}, "
                         f"smoke={sum(r['passed'] for r in model['smoke'].values())}/{len(SMOKE)}")
        lines += [metrics["scope"], f"Metrics and raw logs: {out}"]
        summary = "\n".join(lines) + "\n"
        (out / "summary.txt").write_text(summary)
        print(summary, flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())


# numactl --physcpubind=0-7 --membind=0 \
#   python3 /home/huangxin/code_list/bitnet-moe-script/verify_precision_script/verify_precision_qwen36_bitnet.py \
#     --data /data3/huangxin/model_list/Qwen3.6-35B-A3B/wiki.test.tokens \
#     --threads 8 \
#     --ctx-size 512 \
#     --chunks 32 \
#     --max-chars 262144

# --data             │ 必填，本地 UTF-8 评测文本
# --threads          │ 必填，使用的线程数
# --ctx-size         │ 必填，上下文窗口大小
# --chunks           │ 必填，分块数量
# --max-chars        │ 必填，最大字符数
# 想先快速检查，把 --chunks 32 改为 --chunks 4 即可。三个模型始终使用同一份文本和相同参数。