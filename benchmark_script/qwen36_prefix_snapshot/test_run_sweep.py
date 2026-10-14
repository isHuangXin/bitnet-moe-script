#!/usr/bin/env python3
"""Host-only runner tests; every model, manifest, process and result is a fixture."""
import contextlib
import csv
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("prefix_sweep_runner", Path(__file__).with_name("run_sweep.py"))
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)
ARTIFACT_ROOT = Path("/data2/huangxin/qwen36_cache_experiments")
FIXTURE_PARENT = ARTIFACT_ROOT / "tests"
PP = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65535, 131077, 262144]
REAL_RUN = subprocess.run


def tsv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle, delimiter="\t"))


def write_tsv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(rows)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        FIXTURE_PARENT.mkdir(parents=True, exist_ok=True)
        self.fixture = Path(tempfile.mkdtemp(prefix="sweep-runner.", dir=FIXTURE_PARENT))
        self.root = self.fixture / "artifacts"
        (self.root / "bin").mkdir(parents=True)
        self.probe = self.root / "bin/qwen36-prefix-snapshot"
        self.probe.write_text("fixture probe, not a real model executable\n")
        self.probe.chmod(0o700)
        self.source = self.fixture / "mock-source.hpp"
        self.source.write_text("fixture source\n")
        self.manifest = self.root / "bin/runtime.sha256"
        self.manifest.write_text("".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}\n"
                                         for path in (self.probe, self.source)))
        self.bf, self.q4 = self.fixture / "mock BF16.gguf", self.fixture / "mock Q4_0.gguf"
        self.bf.write_text("NOT WEIGHTS: BF16 fixture\n")
        self.q4.write_text("NOT WEIGHTS: Q4_0 fixture\n")
        self.commands, self.calls, self.hashes = [], [], []
        self.runtime_checks = 0
        self.after_probe = self.on_check = self.mutate_native = None
        self.fail_at = self.hash_change = None
        self.bad_head = self.dirty = self.extra_context = self.real_noise = False
        for key, value in (("ROOT", self.root), ("MANIFEST", self.manifest),
                           ("LLAMA_DIR", self.fixture / "llama"), ("LIB_DIR", self.root / "bin")):
            patcher = mock.patch.object(runner, key, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(runner.subprocess, "run", side_effect=self.fake_process)
        patcher.start()
        self.addCleanup(patcher.stop)

    def argv(self, lengths=PP, **updates):
        values = {"probe": str(self.probe), "bf16-model": str(self.bf), "q4-model": str(self.q4),
                  "threads": "8", "gpu-id": "3", "pp-lengths": ",".join(map(str, lengths)),
                  "repetitions": "3", "warmup": "1"}
        values.update(updates)
        return [item for key, value in values.items() for item in ("--" + key, str(value))]

    def invoke(self, lengths=PP, **updates):
        stdout, stderr = io.StringIO(), io.StringIO()
        before = set((self.root / "runs").glob("sweep.*"))
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = runner.main(self.argv(lengths, **updates))
        sweeps = list(set((self.root / "runs").glob("sweep.*")) - before)
        return code, stdout.getvalue(), stderr.getvalue(), sweeps[0] if len(sweeps) == 1 else None

    def native_files(self, options):
        directory = Path(options["--output-dir"])
        pp = int(options["--prefix-tokens"]) + 1
        repetitions = options["--repetitions"]
        rows = [list(runner.NATIVE_FIELDS)]
        for mode_index, mode in enumerate(runner.MODES):
            for metric_index, (metric, unit) in enumerate(runner.METRICS.items()):
                na = metric in runner.NA_METRICS
                mean = "NA" if na else str(10 * (mode_index + 1) + metric_index + 0.125)
                deviation = "NA" if na else ("0" if repetitions == "1" else "0.25")
                rows.append([mode, metric, unit, repetitions, repetitions,
                             "not_applicable" if na else "pass", mean, deviation])
        write_tsv(directory / "summary.tsv", rows)
        ctx = min(262144, int(options["--ctx-size"]) + (256 if self.extra_context else 0))
        metadata = {
            "benchmark_status": "complete", "model_parity": "passed_for_this_case_and_recorded_trials_only",
            "test": f"pp{pp}", "workload": "prefill-only", "prompt_tokens": str(pp),
            "sampled_outputs_per_request": "1", "decode_forwards_per_request": "0",
            "repetitions_per_mode": repetitions, "warmup_per_mode": options["--warmup"],
            "model": options["--model"], "model_sha256": options["--model-sha256"],
            "runtime_id": options["--runtime-id"], "cuda_visible_devices": "3", "context_actual": str(ctx),
        }
        write_tsv(directory / "metadata.tsv", [["key", "value"], *metadata.items()])
        for name in ("results.tsv", "continuations.tsv"):
            (directory / name).write_text("MOCK ONLY\n")
        (directory / "cache").mkdir()

    def fake_process(self, command, **kwargs):
        command = list(map(str, command))
        self.commands.append(command)
        if command[0] == "git":
            if "rev-parse" in command:
                self.runtime_checks += 1
                if self.on_check:
                    self.on_check(self.runtime_checks)
                text = "bad-revision" if self.bad_head else runner.EXPECTED_COMMIT
            else:
                text = " M tracked.cpp" if self.dirty else ""
            return subprocess.CompletedProcess(command, 0, text + "\n", "")
        if command[0] == "sha256sum":
            if command[1] == "--":
                path = Path(command[2])
                self.assertTrue(path.is_relative_to(self.fixture), "never hash real model weights")
                self.hashes.append(path)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if self.hash_change == path:
                    path.write_bytes(path.read_bytes() + b"changed during hash")
                return subprocess.CompletedProcess(command, 0, f"{digest}  {path}\n", "")
            self.assertEqual(command[1:], ["--check", "--strict", "--quiet", str(self.manifest)])
            for line in self.manifest.read_text().splitlines():
                digest, path = line.split("  ", 1)
                if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                    raise subprocess.CalledProcessError(1, command, stderr="fixture checksum mismatch")
            return subprocess.CompletedProcess(command, 0, "", "")
        self.assertEqual(command[0], str(self.probe))
        options = dict(zip(command[1::2], command[2::2]))
        self.calls.append(options)
        directory = Path(options["--output-dir"])
        self.assertEqual(list(directory.iterdir()), [], "case must be empty before probe")
        self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
        self.assertEqual(directory.stat().st_uid, os.getuid())
        self.assertFalse(directory.is_symlink())
        log_path = Path(kwargs["stdout"].name) if isinstance(kwargs["stdout"].name, str) else None
        self.assertIsNone(log_path)  # fdopen uses an exclusive descriptor, not a case-local file.
        self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "3")
        self.assertTrue(kwargs["env"]["LD_LIBRARY_PATH"].startswith(str(runner.LIB_DIR)))
        self.assertEqual(tsv(directory.parents[1] / "index.tsv")[-1][0], "started", "index was flushed")
        self.native_files(options)
        if self.mutate_native:
            self.mutate_native(directory, len(self.calls))
        if self.real_noise:
            result = REAL_RUN([sys.executable, "-I", "-B", "-c",
                               "import sys; print('NATIVE TABLE stdout'); print('NATIVE stderr', file=sys.stderr)"],
                              stdout=kwargs["stdout"], stderr=kwargs["stderr"], env=kwargs["env"], cwd=kwargs["cwd"])
            self.assertEqual(result.returncode, 0)
        else:
            kwargs["stdout"].write("NATIVE TABLE stdout\nNATIVE stderr\n")
            kwargs["stdout"].flush()
        if self.after_probe:
            self.after_probe(len(self.calls), options)
        return subprocess.CompletedProcess(command, 7 if self.fail_at == len(self.calls) else 0)

    def case_fixture(self, repetitions="3"):
        args = runner.parse_args(self.argv([128], repetitions=repetitions))
        directory = Path(tempfile.mkdtemp(prefix="native.", dir=self.fixture))
        model = runner.Model("BF16", self.bf.resolve(), "a" * 64, runner.file_identity(self.bf))
        options = {"--output-dir": str(directory), "--prefix-tokens": "127", "--ctx-size": "256",
                   "--repetitions": repetitions, "--warmup": "1", "--model": str(model.path),
                   "--model-sha256": model.digest, "--runtime-id": "b" * 64}
        self.native_files(options)
        return directory, model, args

    def read_case(self, fixture):
        directory, model, args = fixture
        return runner.read_case(directory, model, 128, args, "b" * 64)

    def test_exact_sweep_table_and_capture(self):
        self.extra_context = True
        code, out, err, sweep = self.invoke()
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(out.count(runner.TABLE_HEADER), 1)
        self.assertNotIn("NATIVE", out + err)
        self.assertNotIn("Done", out)
        body = [line for line in out.splitlines() if line.split("|", 1)[0].strip() in {"BF16", "Q4_0"}]
        self.assertEqual(len(body), 72)
        self.assertEqual([cell.strip() for cell in runner.TABLE_HEADER.split("|")],
                         ["model", "test", "cache", "n", "TTFT ms", "prefill eval t/s", "effective prompt t/s"])
        separators = [i for i, char in enumerate(runner.TABLE_HEADER) if char == "|"]
        for line in body:
            self.assertEqual([i for i, char in enumerate(line) if char == "|"], separators)
            for cell in line.split("|")[4:]:
                self.assertRegex(cell.strip(), r"^\d+\.\d{2} \+/- \d+\.\d{2}$")
        self.assertEqual(body[0].split("|")[4].strip(), "10.12 +/- 0.25")
        self.assertEqual(len(self.calls), 24)
        self.assertEqual(self.hashes.count(self.bf), 1)
        self.assertEqual(self.hashes.count(self.q4), 1)
        self.assertEqual(self.runtime_checks, 2 + 2 * 24)
        wide = tsv(sweep / "summary.tsv")
        self.assertEqual(wide[0], list(runner.WIDE_FIELDS))
        self.assertEqual(len(wide), 73)
        for i, options in enumerate(self.calls):
            pp = PP[i % 12]
            name, model = ("BF16", self.bf) if i < 12 else ("Q4_0", self.q4)
            self.assertEqual(options["--model"], str(model.resolve()))
            self.assertEqual([options[k] for k in ("--prefix-tokens", "--suffix-tokens", "--decode-tokens", "--ctx-size")],
                             list(map(str, (pp - 1, 1, 0, (pp + 255) // 256 * 256))))
            self.assertEqual([options[k] for k in ("--threads", "--repetitions", "--warmup", "--batch-size", "--ubatch-size", "--seed", "--max-blob-mib", "--max-store-mib", "--disk-reserve-mib")],
                             ["8", "3", "1", "2048", "512", "1", "8192", "16384", "16384"])
            case = sweep / "cases" / f"{name}-pp{pp}"
            self.assertEqual(options["--output-dir"], str(case))
            self.assertEqual(options["--model-sha256"], hashlib.sha256(model.read_bytes()).hexdigest())
            log = sweep / "logs" / f"{name}-pp{pp}.log"
            self.assertEqual(stat.S_IMODE(log.stat().st_mode), 0o600)
            self.assertIn("NATIVE stderr", log.read_text())
            for j, mode in enumerate(runner.MODES):
                row = dict(zip(wide[0], wide[1 + i * 3 + j]))
                self.assertEqual((row["model"], row["test"], row["cache"], row["n"]), (name, f"pp{pp}", mode, "3"))
                self.assertEqual(row["ttft_mean_ms"], str(10 * (j + 1) + 0.125))
                self.assertEqual(row["actual_prefill_mean_tps"], str(10 * (j + 1) + 1.125))
                self.assertEqual(row["effective_prefill_mean_tps"], str(10 * (j + 1) + 2.125))
                self.assertEqual(row["ctx"], str(min(262144, int(options["--ctx-size"]) + 256)))
        events = tsv(sweep / "index.tsv")
        self.assertEqual([row[0] for row in events[1:]], ["started", "pass"] * 24)
        self.assertEqual(stat.S_IMODE(sweep.stat().st_mode), 0o700)
        self.assertFalse(list(sweep.glob(".summary.*")))

    def test_real_child_noise_is_captured(self):
        self.real_noise = True
        code, out, err, sweep = self.invoke([128])
        self.assertEqual((code, err), (0, ""))
        self.assertNotIn("NATIVE", out)
        self.assertEqual(len(tsv(sweep / "summary.tsv")), 7)
        for path in (sweep / "logs").iterdir():
            self.assertIn("NATIVE TABLE stdout", path.read_text())
            self.assertIn("NATIVE stderr", path.read_text())

    def test_cli_rejects_before_work(self):
        invalid = [("threads", value) for value in ("0", "-1", "", "01", "2147483648", "999999999999999999999")]
        invalid += [("gpu-id", value) for value in ("-1", "0,1", "2147483648")]
        invalid += [(key, value) for key in ("repetitions", "warmup") for value in ("0", "1001", "-1", "1.5")]
        invalid += [("pp-lengths", value) for value in ("", "1", "262145", "128,128", "128,", "128,,256", "-2", "1e3")]
        for key, value in invalid:
            with self.subTest(key=key, value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.main(self.argv(**{key: value}))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.main(self.argv() + ["--old-model", "unused"])
        self.assertEqual(self.commands, [])
        self.assertFalse((self.root / "runs").exists())
        valid = runner.parse_args(self.argv([2, 262144], repetitions="1000", warmup="1000", **{"gpu-id": "0"}))
        self.assertEqual((valid.pp_lengths, valid.repetitions, valid.warmup, valid.gpu_id), ([2, 262144], 1000, 1000, 0))

    def test_fixed_probe_and_manifest_coverage(self):
        other = self.fixture / "other-probe"
        other.write_text("not fingerprinted"); other.chmod(0o700)
        code, _, _, _ = self.invoke([128], probe=str(other))
        self.assertNotEqual(code, 0)
        self.manifest.write_text(f"{hashlib.sha256(self.source.read_bytes()).hexdigest()}  {self.source}\n")
        code, _, _, _ = self.invoke([128])
        self.assertNotEqual(code, 0)
        self.assertEqual(self.hashes, [])
        self.assertFalse((self.root / "runs").exists())

    def test_missing_model_and_hash_mutation(self):
        self.q4.write_bytes(b"")
        code, out, _, _ = self.invoke([128])
        self.assertNotEqual(code, 0)
        self.assertEqual(self.hashes, [])
        self.q4.write_text("restored tiny fixture")
        self.hash_change = self.bf
        code, out, err, _ = self.invoke([128])
        self.assertNotEqual(code, 0)
        self.assertIn("changed while hashing", err)
        self.assertNotIn(runner.TABLE_HEADER, out)
        self.assertFalse((self.root / "runs").exists())

    def test_stale_source_and_git_guards(self):
        original = self.source.read_bytes()
        self.source.write_bytes(original + b"stale")
        code, out, _, _ = self.invoke([128])
        self.assertNotEqual(code, 0)
        self.source.write_bytes(original)
        for flag in ("bad_head", "dirty"):
            setattr(self, flag, True)
            code, out, _, _ = self.invoke([128])
            self.assertNotEqual(code, 0)
            setattr(self, flag, False)
        self.assertEqual(self.calls, [])
        self.assertNotIn(self.bf, self.hashes)
        self.assertNotIn(runner.TABLE_HEADER, out)
        self.assertFalse((self.root / "runs").exists())

    def test_native_summary_rejection_matrix(self):
        mutations = {
            "missing": lambda rows: rows.pop(),
            "duplicate": lambda rows: rows.append(rows[1][:]),
            "unknown mode": lambda rows: rows[1].__setitem__(0, "other"),
            "unknown metric": lambda rows: rows[1].__setitem__(1, "other"),
            "wrong unit": lambda rows: rows[1].__setitem__(2, "s"),
            "expected count": lambda rows: rows[1].__setitem__(3, "2"),
            "observed count": lambda rows: rows[1].__setitem__(4, "2"),
            "failed": lambda rows: rows[1].__setitem__(5, "failed"),
            "nan": lambda rows: rows[1].__setitem__(6, "nan"),
            "infinity": lambda rows: rows[1].__setitem__(6, "inf"),
            "zero latency": lambda rows: rows[1].__setitem__(6, "0"),
            "negative mean": lambda rows: rows[1].__setitem__(6, "-1"),
            "negative std": lambda rows: rows[1].__setitem__(7, "-1"),
            "nonfinite std": lambda rows: rows[1].__setitem__(7, "nan"),
            "bad hidden metric": lambda rows: rows[-1].__setitem__(6, "inf"),
            "extra column": lambda rows: rows[1].append("extra"),
            "wrong header": lambda rows: rows[0].__setitem__(3, "expected"),
            "NA wrong status": lambda rows: rows[4].__setitem__(5, "pass"),
            "NA numeric mean": lambda rows: rows[4].__setitem__(6, "0"),
            "NA numeric std": lambda rows: rows[5].__setitem__(7, "0"),
        }
        for name, change in mutations.items():
            fixture = self.case_fixture()
            path = fixture[0] / "summary.tsv"
            data = tsv(path); change(data); write_tsv(path, data)
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.read_case(fixture)
        fixture = self.case_fixture("1")
        path = fixture[0] / "summary.tsv"
        data = tsv(path); data[1][7] = "1"; write_tsv(path, data)
        with self.assertRaises(ValueError):
            self.read_case(fixture)

    def test_metadata_rejection_matrix(self):
        fixture = self.case_fixture()
        keys = [row[0] for row in tsv(fixture[0] / "metadata.tsv")[1:]]
        for key in keys:
            fixture = self.case_fixture()
            path = fixture[0] / "metadata.tsv"
            data = tsv(path)
            for row in data[1:]:
                if row[0] == key:
                    row[1] = "wrong"
            write_tsv(path, data)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.read_case(fixture)
        for kind in ("duplicate", "missing", "header", "columns", "small-context", "large-context"):
            fixture = self.case_fixture()
            path = fixture[0] / "metadata.tsv"; data = tsv(path)
            if kind == "duplicate": data.append(data[1][:])
            elif kind == "missing": data.pop(1)
            elif kind == "header": data[0] = ["name", "value"]
            elif kind == "columns": data[1].append("extra")
            else: data[-1][1] = "127" if kind == "small-context" else "262145"
            write_tsv(path, data)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.read_case(fixture)

    def test_zero_optional_host_times_and_actual_context(self):
        fixture = self.case_fixture()
        path = fixture[0] / "summary.tsv"; data = tsv(path)
        for row in data[1:]:
            if row[1] in ("host_validation_ms", "parity_ms"): row[6] = "0"
        write_tsv(path, data)
        path = fixture[0] / "metadata.tsv"; data = tsv(path); data[-1][1] = "512"; write_tsv(path, data)
        self.assertEqual([row["ctx"] for row in self.read_case(fixture)], ["512"] * 3)

    def assert_partial_failure(self, result, completed):
        code, out, err, sweep = result
        self.assertNotEqual(code, 0)
        self.assertIn("FAILED", err)
        self.assertIn("log:", err)
        self.assertEqual(out.count(runner.TABLE_HEADER), 1)
        events = tsv(sweep / "index.tsv")
        self.assertEqual([row[0] for row in events[1:]], ["started", "pass"] * completed + ["started", "failed"])
        summary = sweep / "summary.tsv"
        if completed:
            self.assertEqual(len(tsv(summary)), 1 + completed * 3)
        else:
            self.assertFalse(summary.exists())
        self.assertEqual(len([line for line in out.splitlines() if line.split("|", 1)[0].strip() in {"BF16", "Q4_0"}]), completed * 3)

    def test_probe_failure_retains_verified_cases(self):
        self.fail_at = 4
        result = self.invoke()
        self.assert_partial_failure(result, 3)
        self.assertEqual(len(self.calls), 4)
        self.assertIn("BF16 pp1024", result[2])
        self.assertTrue((result[3] / "cases/BF16-pp1024/summary.tsv").exists())

    def test_parity_failure_not_published(self):
        def corrupt(directory, count):
            if count == 2:
                path = directory / "metadata.tsv"; data = tsv(path)
                for row in data:
                    if row[0] == "model_parity": row[1] = "failed"
                write_tsv(path, data)
        self.mutate_native = corrupt
        result = self.invoke()
        self.assert_partial_failure(result, 1)
        self.assertEqual(len(self.calls), 2)

    def test_first_incomplete_summary_has_no_wide_rows(self):
        def corrupt(directory, count):
            path = directory / "summary.tsv"; data = tsv(path); data.pop(); write_tsv(path, data)
        self.mutate_native = corrupt
        self.assert_partial_failure(self.invoke(), 0)
        self.assertEqual(len(self.calls), 1)

    def test_pre_case_identity_check(self):
        def change(check):
            if check == 5:  # prep twice, case0 before/after, then case1 before.
                self.q4.write_bytes(self.q4.read_bytes() + b"changed")
        self.on_check = change
        result = self.invoke()
        self.assert_partial_failure(result, 1)
        self.assertEqual(len(self.calls), 1)

    def test_after_last_manifest_change_blocks_last_rows(self):
        def change(count, options):
            if count == 24: self.manifest.write_bytes(self.manifest.read_bytes() + b"\n")
        self.after_probe = change
        result = self.invoke()
        self.assert_partial_failure(result, 23)
        self.assertEqual(len(self.calls), 24)
        self.assertIn("Runtime manifest changed", result[2])
        self.assertIn("Q4_0 pp262144", result[2])

    def test_after_last_model_change_blocks_last_rows(self):
        def change(count, options):
            if count == 24: self.bf.write_bytes(self.bf.read_bytes() + b"changed")
        self.after_probe = change
        result = self.invoke()
        self.assert_partial_failure(result, 23)
        self.assertEqual(len(self.calls), 24)
        self.assertIn("Model changed during sweep", result[2])

    def test_missing_case_summary_retains_prior_rows(self):
        def hide(directory, count):
            if count == 2:
                (directory / "summary.tsv").rename(directory / "summary.fixture-saved.tsv")
        self.mutate_native = hide
        self.assert_partial_failure(self.invoke(), 1)
        self.assertEqual(len(self.calls), 2)

    def test_after_last_library_source_checksum_change(self):
        def change(count, options):
            if count == 24: self.source.write_bytes(self.source.read_bytes() + b"changed")
        self.after_probe = change
        self.assert_partial_failure(self.invoke(), 23)
        self.assertEqual(len(self.calls), 24)

    def test_historical_long_table_is_untouched(self):
        historical = self.root / "runs/sweep.historical"
        historical.mkdir(parents=True)
        path = historical / "summary.tsv"
        old = "model\tpp\tmetric\tmean\nold\t128\tttft_ms\t1\n"
        path.write_text(old)
        code, _, err, sweep = self.invoke([128])
        self.assertEqual((code, err), (0, ""))
        self.assertNotEqual(sweep, historical)
        self.assertEqual(path.read_text(), old)
        self.assertEqual(tsv(sweep / "summary.tsv")[0], list(runner.WIDE_FIELDS))

    def test_atomic_publication_failure_preserves_previous_table(self):
        replace = os.replace
        replacements = []
        def fail_second(source, target):
            replacements.append((source, target))
            self.assertEqual(Path(source).parent, Path(target).parent)
            self.assertEqual(stat.S_IMODE(Path(source).stat().st_mode), 0o600)
            if len(replacements) == 2: raise OSError("mock atomic publication failure")
            replace(source, target)
        with mock.patch.object(runner.os, "replace", side_effect=fail_second):
            self.assert_partial_failure(self.invoke(), 1)
        self.assertEqual(len(replacements), 2)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise SystemExit("Usage: python3 -I -B test_run_sweep.py [fixture_parent]")
    FIXTURE_PARENT = Path(sys.argv[1]) if len(sys.argv) == 2 else FIXTURE_PARENT
    FIXTURE_PARENT = FIXTURE_PARENT.resolve()
    if not FIXTURE_PARENT.is_relative_to(ARTIFACT_ROOT.resolve()):
        raise SystemExit("Fixture parent must be under /data2/huangxin/qwen36_cache_experiments")
    print(f"Host-only runner fixtures: {FIXTURE_PARENT}", flush=True)
    unittest.main(argv=[sys.argv[0]], verbosity=2)
