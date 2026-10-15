#include "snapshot_store.hpp"
#include "benchmark_stats.hpp"
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;
namespace bench = qwen_cache::bench;
using Clock = std::chrono::steady_clock;
using qwen_cache::Identity;
using qwen_cache::Limits;
using qwen_cache::LookupStatus;
using qwen_cache::Snapshot;

struct Options {
    std::string model;
    std::string model_sha256;
    std::string runtime_id;
    fs::path output_dir;
    uint32_t prefix_tokens = 127;
    uint32_t suffix_tokens = 1;
    uint32_t decode_tokens = 8;
    uint32_t ctx_size = 256;
    uint32_t threads = 8;
    uint32_t batch_size = 2048;
    uint32_t ubatch_size = 512;
    uint32_t seed = 1;
    uint32_t repetitions = 3;
    uint32_t warmup = 1;
    Limits limits;
};

static void usage() {
    std::cout
        << "Qwen3.6 cross-request prefix-state probe (not live KV paging)\n"
        << "Usage: qwen36-prefix-snapshot --model PATH --model-sha256 HEX64\n"
        << "       --runtime-id HEX64 --output-dir EMPTY_PRIVATE_DIR [options]\n"
        << "Options:\n"
        << "  --prefix-tokens 127  --suffix-tokens 1  --decode-tokens 8\n"
        << "  --decode-tokens 0: prefill-only; sample one first token for TTFT, no decode forwards\n"
        << "  --ctx-size 256      --threads 8        --seed 1\n"
        << "  --batch-size 2048   --ubatch-size 512\n"
        << "  --repetitions 3    --warmup 1        (each 1..1000, per mode)\n"
        << "  --max-blob-mib 8192 --max-store-mib 16384 --disk-reserve-mib 16384\n"
        << "  --help\n"
        << "Normal execution loads a model and runs recompute/DRAM/io-uring-direct tests.\n"
        << "Set CUDA_VISIBLE_DEVICES to exactly one authorized GPU; internal main_gpu=0.\n"
        << "O_DIRECT bypasses OS page cache; this is NOT a raw-SSD latency measurement.\n"
        << "Help and argument validation do not initialize llama/CUDA or load a model.\n";
}

static uint32_t number(const std::string & text, const std::string & name, bool allow_zero = false) {
    if (text.empty() || text.size() > 10 ||
        !std::all_of(text.begin(), text.end(), [](char c) { return c >= '0' && c <= '9'; })) {
        throw std::runtime_error(name + " must be an unsigned integer");
    }
    const auto value = std::stoull(text);
    if ((!allow_zero && value == 0) || value > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error(name + " is out of range");
    }
    return static_cast<uint32_t>(value);
}

static bool is_digest(const std::string & text) {
    return text.size() == 64 && std::all_of(text.begin(), text.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

static Options parse_options(int argc, char ** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (i + 1 == argc) {
            throw std::runtime_error("Missing value for " + key);
        }
        const std::string value = argv[++i];
        if (key == "--model") o.model = value;
        else if (key == "--model-sha256") o.model_sha256 = value;
        else if (key == "--runtime-id") o.runtime_id = value;
        else if (key == "--output-dir") o.output_dir = value;
        else if (key == "--prefix-tokens") o.prefix_tokens = number(value, key);
        else if (key == "--suffix-tokens") o.suffix_tokens = number(value, key);
        else if (key == "--decode-tokens") o.decode_tokens = number(value, key, true);
        else if (key == "--ctx-size") o.ctx_size = number(value, key);
        else if (key == "--threads") o.threads = number(value, key);
        else if (key == "--batch-size") o.batch_size = number(value, key);
        else if (key == "--ubatch-size") o.ubatch_size = number(value, key);
        else if (key == "--seed") o.seed = number(value, key, true);
        else if (key == "--repetitions") o.repetitions = number(value, key, true);
        else if (key == "--warmup") o.warmup = number(value, key, true);
        else if (key == "--max-blob-mib") o.limits.max_blob_bytes = uint64_t(number(value, key)) << 20;
        else if (key == "--max-store-mib") o.limits.max_store_bytes = uint64_t(number(value, key)) << 20;
        else if (key == "--disk-reserve-mib") o.limits.disk_reserve_bytes = uint64_t(number(value, key)) << 20;
        else throw std::runtime_error("Unknown option: " + key);
    }
    bench::validate_counts(o.repetitions, o.warmup);
    if (o.model.empty() || o.output_dir.empty() || !is_digest(o.model_sha256) || !is_digest(o.runtime_id)) {
        throw std::runtime_error("Required: --model, --model-sha256 HEX64, --runtime-id HEX64, --output-dir");
    }
    const uint64_t total = uint64_t(o.prefix_tokens) + o.suffix_tokens + o.decode_tokens;
    if (total > o.ctx_size || o.ctx_size > 262144) {
        throw std::runtime_error("Require prefix + suffix + decode <= context <= 262144; no silent truncation");
    }
    if (o.ubatch_size > o.batch_size || o.limits.max_blob_bytes > o.limits.max_store_bytes) {
        throw std::runtime_error("Require ubatch <= batch and max-blob <= max-store");
    }
    return o;
}

static std::string field(const std::string & value) {
    std::string result;
    for (char c : value) {
        if (c == '\\') result += "\\\\";
        else if (c == '\t') result += "\\t";
        else if (c == '\n') result += "\\n";
        else if (c == '\r') result += "\\r";
        else result += c;
    }
    return result;
}

class Table {
    std::unique_ptr<FILE, decltype(&std::fclose)> stream_{nullptr, &std::fclose};
public:
    explicit Table(const fs::path & path) {
        const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
        if (fd < 0) throw std::runtime_error("Cannot exclusively create " + path.string() + ": " + std::strerror(errno));
        FILE * stream = ::fdopen(fd, "w");
        if (!stream) {
            ::close(fd);
            throw std::runtime_error("fdopen failed for " + path.string());
        }
        stream_.reset(stream);
    }
    void row(const std::vector<std::string> & values) {
        std::string line;
        for (size_t i = 0; i < values.size(); ++i) {
            if (i) line += '\t';
            line += field(values[i]);
        }
        line += '\n';
        if (std::fwrite(line.data(), 1, line.size(), stream_.get()) != line.size() || std::fflush(stream_.get()) != 0) {
            throw std::runtime_error("Failed to write experiment results");
        }
    }
};

static void check_output_dir(const fs::path & path) {
    struct stat info {};
    if (::lstat(path.c_str(), &info) != 0 || !S_ISDIR(info.st_mode) || info.st_uid != ::geteuid() ||
        (info.st_mode & 0077) != 0 || !fs::is_empty(path)) {
        throw std::runtime_error("Output directory must already exist, be private/owned, non-symlink, and empty");
    }
}

static double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

template <class F> static double host_time(F && work) {
    const auto start = Clock::now();
    work();
    return elapsed_ms(start);
}

template <class F> static double gpu_time(llama_context * ctx, F && work) {
    llama_synchronize(ctx);
    const auto start = Clock::now();
    work();
    llama_synchronize(ctx);
    return elapsed_ms(start);
}

static long peak_rss_kib() {
    struct rusage usage {};
    if (::getrusage(RUSAGE_SELF, &usage) != 0) return -1;
    return usage.ru_maxrss;
}

static std::string metric(double value) {
    if (!std::isfinite(value)) return "NA";
    std::ostringstream out;
    out << std::setprecision(10) << value;
    return out.str();
}

struct Result {
    std::string mode = "startup";
    std::string status = "error";
    std::string detail;
    std::optional<bench::Trial> trial;
    std::optional<bench::Rates> rates;
    uint64_t reused_tokens = 0;
    uint64_t state_bytes = 0;
    uint64_t stored_file_bytes = 0;
    uint64_t evaluated_prefill_tokens = 0;
    uint64_t evaluated_decode_tokens = 0;
    uint64_t sampled_output_count = 0;
    double reset_ms = NAN;
    double prefix_ms = NAN;
    double tail_ms = NAN;
    double capture_reset_ms = NAN;
    double capture_prefix_ms = NAN;
    double export_ms = NAN;
    double store_ms = NAN;
    double interference_ms = NAN;
    double load_ms = NAN;
    double restore_ms = NAN;
    double first_sample_ms = NAN;
    double decode_ms = NAN;
    double ttft_ms = NAN;
    double request_ms = NAN;
    double prompt_ready_ms = NAN;
    double last_sample_ms = NAN;
    double decode_forward_ms = NAN;
    double host_validation_ms = NAN;
    double parity_ms = NAN;
    double max_abs_error = NAN;
    double max_rel_error = NAN;
    bool tokens_equal = false;
};

static std::string optional_metric(const std::optional<double> & value) {
    return value ? metric(*value) : "NA";
}

static void result_header(Table & table) {
    table.row({"mode", "status", "prefix_tokens", "suffix_tokens", "decode_tokens", "reused_tokens",
        "state_bytes", "stored_file_bytes", "reset_ms", "prefix_ms", "tail_ms", "capture_reset_ms",
        "capture_prefix_ms", "export_ms", "store_ms", "interference_ms", "load_validation_ms",
        "restore_ms", "first_sample_ms", "decode_ms", "ttft_ms", "request_ms", "max_abs_logit_error",
        "max_rel_logit_error", "tokens_equal", "process_peak_rss_kib", "detail", "phase", "trial", "order",
        "prompt_ready_ms", "last_sample_ms", "decode_forward_ms", "host_validation_ms", "parity_ms",
        "evaluated_prefill_tokens", "evaluated_decode_tokens", "actual_prefill_eval_tps",
        "effective_prompt_tps", "conditioned_decode_forward_tps", "generation_after_first_tps",
        "end_to_end_output_tps", "test", "sampled_output_count"});
}

static void write_result(Table & table, const Result & r, const Options & o) {
    table.row({r.mode, r.status, std::to_string(o.prefix_tokens), std::to_string(o.suffix_tokens),
        std::to_string(o.decode_tokens), std::to_string(r.reused_tokens), std::to_string(r.state_bytes),
        std::to_string(r.stored_file_bytes), metric(r.reset_ms), metric(r.prefix_ms), metric(r.tail_ms),
        metric(r.capture_reset_ms), metric(r.capture_prefix_ms), metric(r.export_ms), metric(r.store_ms),
        metric(r.interference_ms), metric(r.load_ms), metric(r.restore_ms), metric(r.first_sample_ms),
        metric(r.decode_ms), metric(r.ttft_ms), metric(r.request_ms), metric(r.max_abs_error),
        metric(r.max_rel_error), r.tokens_equal ? "true" : "false", std::to_string(peak_rss_kib()), r.detail,
        r.trial ? bench::phase_name(r.trial->phase) : "setup",
        r.trial ? std::to_string(r.trial->trial) : "NA", r.trial ? std::to_string(r.trial->order) : "NA",
        metric(r.prompt_ready_ms), metric(r.last_sample_ms), metric(r.decode_forward_ms),
        metric(r.host_validation_ms), metric(r.parity_ms), std::to_string(r.evaluated_prefill_tokens),
        std::to_string(r.evaluated_decode_tokens), r.rates ? metric(r.rates->actual_prefill_eval_tps) : "NA",
        r.rates ? metric(r.rates->effective_prompt_tps) : "NA",
        r.rates ? optional_metric(r.rates->conditioned_decode_forward_tps) : "NA",
        r.rates ? optional_metric(r.rates->generation_after_first_tps) : "NA",
        r.rates ? metric(r.rates->end_to_end_output_tps) : "NA",
        "pp" + std::to_string(uint64_t(o.prefix_tokens) + o.suffix_tokens), std::to_string(r.sampled_output_count)});
}

struct MetricSpec {
    const char * name;
    const char * unit;
    std::function<std::optional<double>(const Result &)> value;
    bool allow_all_na = false;
};

static bool write_summaries(Table & table, const std::vector<Result> & records, const Options & o,
                            bool force_failed = false) {
    const std::vector<MetricSpec> metrics = {
        {"ttft_ms", "ms", [](const Result & r) { return std::optional<double>(r.ttft_ms); }},
        {"actual_prefill_eval_tps", "tokens/s", [](const Result & r) {
            return r.rates ? std::optional<double>(r.rates->actual_prefill_eval_tps) : std::nullopt; }},
        {"effective_prompt_tps", "tokens/s", [](const Result & r) {
            return r.rates ? std::optional<double>(r.rates->effective_prompt_tps) : std::nullopt; }},
        {"conditioned_decode_forward_tps", "tokens/s", [](const Result & r) {
            return r.rates ? r.rates->conditioned_decode_forward_tps : std::nullopt; }, o.decode_tokens == 0},
        {"generation_after_first_tps", "tokens/s", [](const Result & r) {
            return r.rates ? r.rates->generation_after_first_tps : std::nullopt; }, o.decode_tokens <= 1},
        {"end_to_end_output_tps", "tokens/s", [](const Result & r) {
            return r.rates ? std::optional<double>(r.rates->end_to_end_output_tps) : std::nullopt; }},
        {"prompt_ready_ms", "ms", [](const Result & r) { return std::optional<double>(r.prompt_ready_ms); }},
        {"last_sample_ms", "ms", [](const Result & r) { return std::optional<double>(r.last_sample_ms); }},
        {"request_ms", "ms", [](const Result & r) { return std::optional<double>(r.request_ms); }},
        {"host_validation_ms", "ms", [](const Result & r) { return std::optional<double>(r.host_validation_ms); }},
        {"parity_ms", "ms", [](const Result & r) { return std::optional<double>(r.parity_ms); }}
    };
    struct Row { bench::Mode mode; size_t metric_index; bench::Summary summary; };
    std::vector<Row> rows;
    bool complete = !force_failed;
    for (auto mode : {bench::Mode::recompute, bench::Mode::dram, bench::Mode::direct_file}) {
        for (size_t i = 0; i < metrics.size(); ++i) {
            std::vector<bench::Observation> observations;
            for (const auto & r : records) {
                if (r.trial) observations.push_back({*r.trial, r.status == "pass", metrics[i].value(r)});
            }
            auto summary = bench::summarize(mode, o.repetitions, observations, metrics[i].allow_all_na, force_failed);
            complete = complete && (summary.status == "pass" || summary.status == "not_applicable");
            rows.push_back({mode, i, std::move(summary)});
        }
    }
    // A failed/incomplete experiment cannot leave successful-looking partial means.
    if (!complete) {
        for (auto & row : rows) {
            row.summary.status = "failed";
            row.summary.mean.reset();
            row.summary.stddev.reset();
        }
    }
    table.row({"mode", "metric", "unit", "n_expected", "n_observed", "status", "mean", "stddev"});
    for (const auto & row : rows) {
        const auto & s = row.summary;
        const auto & m = metrics[row.metric_index];
        table.row({bench::mode_name(row.mode), m.name, m.unit, std::to_string(s.n_expected),
            std::to_string(s.n_observed), s.status, optional_metric(s.mean), optional_metric(s.stddev)});
    }
    auto cell = [](const bench::Summary & summary) {
        if (summary.status != "pass" || !summary.mean || !summary.stddev) return summary.status;
        std::ostringstream text;
        text << std::fixed << std::setprecision(2) << *summary.mean << " +/- " << *summary.stddev;
        return text.str();
    };
    const bool prefill_only = o.decode_tokens == 0;
    std::cout << "\nInstrumented pp" << uint64_t(o.prefix_tokens) + o.suffix_tokens
              << (prefill_only ? " prefill-only" : " prefix/decode")
              << " statistics: mean +/- sample stddev\n"
              << "P=" << o.prefix_tokens << ", S=" << o.suffix_tokens << ", D=" << o.decode_tokens
              << "; prefill evaluated tokens: recompute=P+S, hits=S\n"
              << std::left << std::setw(16) << "mode" << std::setw(12) << "n/expected"
              << std::setw(27) << "TTFT ms" << std::setw(27) << "prefill eval t/s"
              << std::setw(27) << "effective prompt t/s";
    if (!prefill_only) std::cout << "decode-forward t/s";
    std::cout << '\n';
    const size_t displayed_metrics = prefill_only ? 3 : 4;
    for (auto mode : {bench::Mode::recompute, bench::Mode::dram, bench::Mode::direct_file}) {
        std::vector<const bench::Summary *> selected;
        for (const auto & row : rows) {
            if (row.mode == mode && row.metric_index < displayed_metrics) selected.push_back(&row.summary);
        }
        const auto & count = *selected.front();
        std::cout << std::left << std::setw(16) << bench::mode_name(mode)
                  << std::setw(12) << (std::to_string(count.n_observed) + "/" + std::to_string(count.n_expected));
        for (const auto * s : selected) std::cout << std::setw(27) << cell(*s);
        std::cout << '\n';
    }
    if (prefill_only) {
        std::cout << "Prefill-only: sample one first token for TTFT/parity; zero subsequent decode forwards.\n";
    } else {
        std::cout << "Decode is prefix-conditioned greedy forward work, not independent llama-bench tg.\n";
    }
    std::cout << "Wall latency includes host validation. O_DIRECT bypasses OS page cache; this is not raw SSD latency.\n";
    return complete;
}

struct BackendGuard {
    BackendGuard() { llama_backend_init(); ggml_backend_load_all(); }
    ~BackendGuard() { llama_backend_free(); }
};

class Evaluator {
    llama_context * ctx_;
    llama_batch batch_;
    uint32_t capacity_;
public:
    explicit Evaluator(llama_context * ctx)
        : ctx_(ctx), batch_(llama_batch_init(static_cast<int32_t>(llama_n_batch(ctx)), 0, 1)),
          capacity_(llama_n_batch(ctx)) {
        if (!batch_.token || !batch_.pos || !batch_.logits || !batch_.seq_id || !capacity_) {
            llama_batch_free(batch_);
            throw std::runtime_error("Cannot allocate token batch");
        }
    }
    ~Evaluator() { llama_batch_free(batch_); }
    Evaluator(const Evaluator &) = delete;
    Evaluator & operator=(const Evaluator &) = delete;

    void decode(const std::vector<int32_t> & tokens, size_t offset, size_t count, uint32_t position,
                double * forward_ms = nullptr) {
        if (!count || offset > tokens.size() || count > tokens.size() - offset ||
            uint64_t(position) + count > llama_n_ctx(ctx_)) {
            throw std::runtime_error("Invalid decode range");
        }
        for (size_t done = 0; done < count;) {
            const auto n = static_cast<int32_t>(std::min<size_t>(capacity_, count - done));
            batch_.n_tokens = n;
            for (int32_t i = 0; i < n; ++i) {
                batch_.token[i] = tokens[offset + done + i];
                batch_.pos[i] = static_cast<llama_pos>(position + done + i);
                batch_.n_seq_id[i] = 1;
                batch_.seq_id[i][0] = 0;
                batch_.logits[i] = (i == n - 1);
            }
            int rc = 0;
            if (forward_ms) {
                llama_synchronize(ctx_);
                const auto start = Clock::now();
                rc = llama_decode(ctx_, batch_);
                llama_synchronize(ctx_);
                *forward_ms += elapsed_ms(start);
            } else {
                rc = llama_decode(ctx_, batch_);
            }
            if (rc != 0) throw std::runtime_error("llama_decode failed with status " + std::to_string(rc));
            done += n;
        }
    }
};

static std::vector<int32_t> synthetic_tokens(const llama_vocab * vocab, uint32_t count, uint64_t seed) {
    std::vector<int32_t> eligible;
    for (int32_t id = 0; id < llama_vocab_n_tokens(vocab); ++id) {
        if (!llama_vocab_is_control(vocab, id) && !llama_vocab_is_eog(vocab, id)) eligible.push_back(id);
    }
    if (eligible.size() < 2) throw std::runtime_error("Vocabulary has too few non-control tokens");
    uint64_t state = seed ? seed : UINT64_C(0x2545f4914f6cdd1d);
    std::vector<int32_t> tokens;
    tokens.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        const uint64_t sample = state * UINT64_C(2685821657736338717);
        tokens.push_back(eligible[sample % eligible.size()]);
    }
    if (llama_vocab_get_add_bos(vocab)) {
        const auto bos = llama_vocab_bos(vocab);
        if (bos < 0 || bos >= llama_vocab_n_tokens(vocab)) throw std::runtime_error("Invalid BOS token");
        tokens.front() = bos;
    }
    return tokens;
}

static std::string model_meta(llama_model * model, const char * key) {
    char buffer[256];
    const int32_t n = llama_model_meta_val_str(model, key, buffer, sizeof(buffer));
    if (n < 0 || static_cast<size_t>(n) >= sizeof(buffer)) {
        throw std::runtime_error(std::string("Missing or oversized model metadata: ") + key);
    }
    return buffer;
}

static Identity make_identity(const Options & o, llama_context * ctx, llama_model * model,
                              ggml_backend_dev_t device, const std::vector<int32_t> & tokens) {
    std::ostringstream config;
    auto add = [&](const std::string & key, const std::string & value) {
        config << key.size() << ':' << key << value.size() << ':' << value;
    };
    struct stat info {};
    if (::stat(o.model.c_str(), &info) != 0) throw std::runtime_error("Cannot stat model");
    add("format", "qwen-prefix-snapshot-v1");
    add("model_sha256", o.model_sha256);
    add("model_path", fs::canonical(o.model).string());
    add("model_bytes", std::to_string(info.st_size));
    add("model_mtime_seconds", std::to_string(info.st_mtim.tv_sec));
    add("model_mtime_ns", std::to_string(info.st_mtim.tv_nsec));
    add("runtime_id", o.runtime_id);
    add("architecture", model_meta(model, "general.architecture"));
    add("file_type", model_meta(model, "general.file_type"));
    add("backend", "CUDA");
    add("device", ggml_backend_dev_name(device));
    add("device_description", ggml_backend_dev_description(device));
    add("visibility", std::getenv("CUDA_VISIBLE_DEVICES"));
    add("context_requested", std::to_string(o.ctx_size));
    add("context_actual", std::to_string(llama_n_ctx(ctx)));
    add("batch_actual", std::to_string(llama_n_batch(ctx)));
    add("ubatch_actual", std::to_string(llama_n_ubatch(ctx)));
    add("threads", std::to_string(o.threads));
    add("placement", "ngl=99;split=none;main_gpu=0;kv_offload=1;op_offload=1");
    add("state", "seq=0;flags=NONE;n_seq_max=1;n_rs_seq=0;kv=f16/f16;recurrent=f32");
    add("attention", "flash=auto;causal=model-default;rope=model-default;adapters=none");
    add("sampler", "fresh-greedy;fixed-decode-count;ignore-eog=1");
    add("layout", "matching-build-and-context-only;no-cross-backend-reuse");
    Identity identity;
    identity.configuration = config.str();
    identity.prefix_tokens.assign(tokens.begin(), tokens.begin() + o.prefix_tokens);
    identity.processed_tokens = o.prefix_tokens;
    return identity;
}

struct Continuation {
    std::vector<float> logits;
    std::vector<int32_t> tokens;
};

static Continuation continue_request(const Options & o, llama_context * ctx, Evaluator & evaluator,
                                     const llama_vocab * vocab, Result & result, Clock::time_point request_start) {
    result.prompt_ready_ms = elapsed_ms(request_start);
    result.host_validation_ms = 0;
    result.decode_forward_ms = 0;
    result.evaluated_decode_tokens = 0;
    result.sampled_output_count = 0;
    const auto n_vocab = llama_vocab_n_tokens(vocab);
    auto checked_logits = [&]() -> const float * {
        const float * values = nullptr;
        result.host_validation_ms += host_time([&] {
            values = llama_get_logits_ith(ctx, -1);
            if (!values || n_vocab <= 0 ||
                !std::all_of(values, values + n_vocab, [](float x) { return std::isfinite(x); })) {
                throw std::runtime_error("Missing or non-finite logits during continuation");
            }
        });
        return values;
    };
    const float * logits = checked_logits();
    Continuation output;
    result.host_validation_ms += host_time([&] { output.logits.assign(logits, logits + n_vocab); });
    output.tokens.reserve(std::max<uint32_t>(1, o.decode_tokens));
    auto sampler = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>(
        llama_sampler_chain_init(llama_sampler_chain_default_params()), &llama_sampler_free);
    if (!sampler) throw std::runtime_error("Cannot allocate greedy sampler");
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
    auto sample = [&]() {
        const auto token = llama_sampler_sample(sampler.get(), ctx, -1);
        if (token < 0 || token >= n_vocab) throw std::runtime_error("Invalid generated token");
        return token;
    };
    const auto sample_start = Clock::now();
    int32_t next = sample();
    const auto first_sample = Clock::now();
    auto last_sample = first_sample;
    result.first_sample_ms = std::chrono::duration<double, std::milli>(first_sample - sample_start).count();
    result.ttft_ms = std::chrono::duration<double, std::milli>(first_sample - request_start).count();
    if (o.decode_tokens == 0) {
        // Sampling the existing prefill logits does not consume another context position.
        output.tokens.push_back(next);
        result.sampled_output_count = output.tokens.size();
        result.decode_ms = 0;
        result.last_sample_ms = result.ttft_ms;
        result.request_ms = elapsed_ms(request_start);
        return output;
    }
    result.decode_ms = gpu_time(ctx, [&] {
        std::vector<int32_t> one(1);
        for (uint32_t i = 0; i < o.decode_tokens; ++i) {
            output.tokens.push_back(next);
            one[0] = next;
            evaluator.decode(one, 0, 1, o.prefix_tokens + o.suffix_tokens + i, &result.decode_forward_ms);
            ++result.evaluated_decode_tokens;
            checked_logits();
            if (i + 1 < o.decode_tokens) {
                next = sample();
                last_sample = Clock::now();
            }
        }
    });
    result.last_sample_ms = std::chrono::duration<double, std::milli>(last_sample - request_start).count();
    result.sampled_output_count = output.tokens.size();
    result.request_ms = elapsed_ms(request_start);
    return output;
}

static void write_continuation(Table & table, const Result & result, const Continuation & continuation) {
    for (size_t i = 0; i < continuation.tokens.size(); ++i) {
        table.row({result.mode, std::to_string(i), std::to_string(continuation.tokens[i]),
            result.trial ? bench::phase_name(result.trial->phase) : "setup",
            result.trial ? std::to_string(result.trial->trial) : "NA",
            result.trial ? std::to_string(result.trial->order) : "NA"});
    }
}

static uint64_t stored_bytes(const fs::path & path) {
    uint64_t bytes = 0;
    for (const auto & entry : fs::directory_iterator(path)) {
        if (entry.is_regular_file()) bytes += entry.file_size();
    }
    return bytes;
}

template <class Store>
static void prepare_snapshot(Store & store, const Options & o, llama_context * ctx, Evaluator & evaluator,
                             const Identity & identity, const std::vector<int32_t> & a,
                             Result & setup, const fs::path & file_root = {}) {
    setup.capture_reset_ms = gpu_time(ctx, [&] { llama_memory_clear(llama_get_memory(ctx), true); });
    setup.capture_prefix_ms = gpu_time(ctx, [&] { evaluator.decode(a, 0, o.prefix_tokens, 0); });
    if (llama_memory_seq_pos_max(llama_get_memory(ctx), 0) != static_cast<llama_pos>(o.prefix_tokens - 1)) {
        throw std::runtime_error("Prefix checkpoint position mismatch");
    }
    Snapshot snapshot;
    snapshot.identity = identity;
    setup.export_ms = gpu_time(ctx, [&] {
        const size_t size = llama_state_seq_get_size(ctx, 0);
        snapshot.state = qwen_cache::checked_export(size, [&](uint8_t * data, size_t capacity) {
            return llama_state_seq_get_data(ctx, data, capacity, 0);
        }, o.limits);
    });
    setup.state_bytes = snapshot.state.size();
    setup.store_ms = host_time([&] { store.put("prefix-A", std::move(snapshot)); });
    if (!file_root.empty()) setup.stored_file_bytes = stored_bytes(file_root);
    setup.status = "prepared";
}

static void write_setup(Table & metadata, const Result & setup) {
    const std::string key = "setup." + setup.mode + ".";
    metadata.row({key + "capture_reset_ms", metric(setup.capture_reset_ms)});
    metadata.row({key + "capture_prefix_ms", metric(setup.capture_prefix_ms)});
    metadata.row({key + "export_ms", metric(setup.export_ms)});
    metadata.row({key + "store_ms", metric(setup.store_ms)});
    metadata.row({key + "state_bytes", std::to_string(setup.state_bytes)});
    metadata.row({key + "stored_file_bytes", std::to_string(setup.stored_file_bytes)});
}

static void run_interference(llama_context * ctx, Evaluator & evaluator,
                             const std::vector<int32_t> & b, Result & result) {
    result.interference_ms = gpu_time(ctx, [&] {
        llama_memory_clear(llama_get_memory(ctx), true);
        evaluator.decode(b, 0, b.size(), 0);
    });
}

static Continuation recompute_request(const Options & o, llama_context * ctx, Evaluator & evaluator,
                                      const llama_vocab * vocab, const std::vector<int32_t> & a, Result & result) {
    const auto start = Clock::now();
    result.reset_ms = gpu_time(ctx, [&] { llama_memory_clear(llama_get_memory(ctx), true); });
    result.prefix_ms = gpu_time(ctx, [&] { evaluator.decode(a, 0, o.prefix_tokens, 0); });
    result.evaluated_prefill_tokens = o.prefix_tokens;
    result.tail_ms = gpu_time(ctx, [&] { evaluator.decode(a, o.prefix_tokens, o.suffix_tokens, o.prefix_tokens); });
    result.evaluated_prefill_tokens += o.suffix_tokens;
    return continue_request(o, ctx, evaluator, vocab, result, start);
}

template <class Store>
static Continuation cached_request(Store & store, const Options & o, llama_context * ctx, Evaluator & evaluator,
                                   const llama_vocab * vocab, const Identity & identity,
                                   const std::vector<int32_t> & a, const Result & setup, Result & result) {
    result.state_bytes = setup.state_bytes;
    result.stored_file_bytes = setup.stored_file_bytes;
    auto clear = [&] { llama_memory_clear(llama_get_memory(ctx), true); };
    const auto start = Clock::now();
    qwen_cache::LookupResult loaded;
    result.load_ms = host_time([&] { loaded = store.get("prefix-A", identity); });
    if (loaded.status != LookupStatus::hit || !loaded.snapshot) {
        throw std::runtime_error("Expected stored prefix hit: " + loaded.detail);
    }
    if (loaded.snapshot->state.size() != result.state_bytes) throw std::runtime_error("Stored state size changed");
    result.reset_ms = 0;
    result.restore_ms = 0;
    auto timed_clear = [&] { result.reset_ms += gpu_time(ctx, clear); };
    qwen_cache::checked_restore(loaded.snapshot->state, timed_clear, [&](const uint8_t * data, size_t size) {
        size_t consumed = 0;
        result.restore_ms += gpu_time(ctx, [&] { consumed = llama_state_seq_set_data(ctx, data, size, 0); });
        return consumed;
    });
    if (llama_memory_seq_pos_max(llama_get_memory(ctx), 0) != static_cast<llama_pos>(o.prefix_tokens - 1)) {
        timed_clear();
        throw std::runtime_error("Restored checkpoint position mismatch");
    }
    result.prefix_ms = 0;
    result.tail_ms = gpu_time(ctx, [&] { evaluator.decode(a, o.prefix_tokens, o.suffix_tokens, o.prefix_tokens); });
    result.evaluated_prefill_tokens = o.suffix_tokens;
    return continue_request(o, ctx, evaluator, vocab, result, start);
}

static void derive_metrics(Result & result, const Options & o, bench::Mode mode) {
    const uint64_t expected_prefill = mode == bench::Mode::recompute
        ? uint64_t(o.prefix_tokens) + o.suffix_tokens : o.suffix_tokens;
    const uint64_t expected_samples = std::max<uint32_t>(1, o.decode_tokens);
    if (result.evaluated_prefill_tokens != expected_prefill || result.evaluated_decode_tokens != o.decode_tokens ||
        result.sampled_output_count != expected_samples) {
        throw std::runtime_error("Evaluated/sampled token counters do not match the declared workload");
    }
    result.rates = bench::derive_rates(mode, o.prefix_tokens, o.suffix_tokens, o.decode_tokens,
        {result.prefix_ms, result.tail_ms, result.prompt_ready_ms, result.ttft_ms,
         result.last_sample_ms, result.request_ms, result.decode_forward_ms});
}

static int run(const Options & o) {
    const char * visibility = std::getenv("CUDA_VISIBLE_DEVICES");
    if (!visibility || !*visibility || std::strchr(visibility, ',')) {
        throw std::runtime_error("Set CUDA_VISIBLE_DEVICES to exactly one authorized GPU before a model run");
    }
    if (std::getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY")) {
        throw std::runtime_error("Unset GGML_CUDA_ENABLE_UNIFIED_MEMORY for this explicit GPU/DRAM snapshot baseline");
    }
    if (!fs::is_regular_file(o.model) || fs::file_size(o.model) == 0) throw std::runtime_error("Model file missing or empty");
    check_output_dir(o.output_dir);
    Table results(o.output_dir / "results.tsv");
    Table metadata(o.output_dir / "metadata.tsv");
    Table continuations(o.output_dir / "continuations.tsv");
    Table summaries(o.output_dir / "summary.tsv");
    result_header(results);
    metadata.row({"key", "value"});
    continuations.row({"mode", "index", "token_id", "phase", "trial", "order"});
    metadata.row({"measurement_schema", "qwen-prefix-statistics-v3; prefill-only D=0 supported; snapshot envelope unchanged"});
    metadata.row({"test", "pp" + std::to_string(uint64_t(o.prefix_tokens) + o.suffix_tokens)});
    metadata.row({"prompt_tokens", std::to_string(uint64_t(o.prefix_tokens) + o.suffix_tokens)});
    metadata.row({"workload", o.decode_tokens == 0 ? "prefill-only" : "prefill-and-decode"});
    metadata.row({"sampled_outputs_per_request", std::to_string(std::max<uint32_t>(1, o.decode_tokens))});
    metadata.row({"decode_forwards_per_request", std::to_string(o.decode_tokens)});
    metadata.row({"repetitions_per_mode", std::to_string(o.repetitions)});
    metadata.row({"warmup_per_mode", std::to_string(o.warmup)});
    metadata.row({"threads", std::to_string(o.threads)});
    metadata.row({"experiment", "single-prefix A/B/A; cross-request complete host snapshots; not live KV paging"});
    metadata.row({"file_backend", "io-uring-direct; O_DIRECT bypasses OS page cache; not raw SSD latency"});
    // Storage configuration for the file-snapshot backend only.
    metadata.row({"storage_backend", "ssd"});
    metadata.row({"io_mode", "io_uring+O_DIRECT"});
    metadata.row({"io_queue_depth", std::to_string(qwen_cache::detail::direct_queue_depth)});
    metadata.row({"io_chunk_bytes", std::to_string(qwen_cache::detail::direct_chunk_bytes)});
    metadata.row({"io_alignment_bytes", std::to_string(qwen_cache::detail::direct_alignment)});
    metadata.row({"file_write_mode", "buffered+fsync"});
    metadata.row({"model", fs::canonical(o.model).string()});
    metadata.row({"model_sha256", o.model_sha256});
    metadata.row({"runtime_id", o.runtime_id});
    metadata.row({"cuda_visible_devices", visibility});
    metadata.row({"limits_encoded_blob_bytes", std::to_string(o.limits.max_blob_bytes)});
    metadata.row({"limits_store_bytes", std::to_string(o.limits.max_store_bytes)});
    metadata.row({"disk_reserve_bytes", std::to_string(o.limits.disk_reserve_bytes)});
    metadata.row({"synthetic_tokens", "xorshift64star-v1; uniform modulo ordered non-control/non-EOG token IDs; optional BOS replaces token0"});
    metadata.row({"seed_A", std::to_string(o.seed)});
    const uint64_t seed_b = uint64_t(o.seed) ^ UINT64_C(0x9e3779b97f4a7c15);
    metadata.row({"seed_B", std::to_string(seed_b)});
    metadata.row({"parity_atol", "0.0001"});
    metadata.row({"parity_rtol", "0.0001"});
    metadata.row({"timing_scope", "tokens-ready through first greedy sample/completed fixed-count continuation; excludes model/hash startup and parity comparison"});
    metadata.row({"timing_design", "golden reference excluded; all equal per-mode warmups before measurements; rotating order per phase-local round"});
    metadata.row({"schedule", "mode=(trial+order)%3 over recompute,dram,io-uring-direct; zero-based indices; B interference before every request outside request timing"});
    metadata.row({"statistics", "arithmetic mean of per-trial values; sample stddev n-1; singleton stddev=0; no failed/incomplete sample filtering"});
    metadata.row({"measurement_scope", "instrumented single-length synthetic requests, not server/network latency or raw independent llama-bench pp/tg"});
    metadata.row({"cache_setup_scope", "each store prepared once; capture/export/publication excluded from hit request timing and recorded separately"});
    metadata.row({"ttft_caveat", "includes state reset, lookup/validation/restore and host logit capture/checks; no validation-time subtraction"});
    metadata.row({"throughput_scope", "actual prefill counts only evaluated tokens; effective prompt counts reused prefix too and includes restore; D=0 decode rates are NA"});
    metadata.row({"generation_scope", o.decode_tokens == 0
        ? "one first token sampled from prefill logits; no continuation forward; end-to-end output rate is 1/TTFT, not decode throughput"
        : "post-first generation counts D-1 sample intervals; D=1 is NA; last_sample_ms excludes the final sampled-token forward"});
    metadata.row({"decode_scope", o.decode_tokens == 0
        ? "prefill-only: zero post-prefill llama_decode calls, zero decode duration, no extra context position consumed"
        : "D synchronized forwards, including final sampled-token update; forward timing excludes sampling/validation; decode_ms/request_ms include host checks"});
    metadata.row({"parity_scope", o.decode_tokens == 0
        ? "full post-prefill logits and one sampled first token; no multi-token continuation tested"
        : "full post-suffix logits and fixed-count greedy continuation, finite checks after every evaluated token"});
    metadata.row({"reset_scope", "one zeroing reset per request, timed separately from import; failed imports are cleared again"});
    metadata.row({"file_cache_policy", "O_DIRECT reads bypass OS page cache; device/controller caches uncontrolled; no buffered fallback or raw-SSD latency claim"});
    Result current;
    std::vector<Result> records;
    bool current_recorded = false;
    bool summary_written = false;
    try {
        BackendGuard backend;
        ggml_backend_dev_t devices[2] = {ggml_backend_dev_by_name("CUDA0"), nullptr};
        if (!devices[0] || ggml_backend_dev_by_name("CUDA1")) {
            throw std::runtime_error("Expected exactly one CUDA-visible device named CUDA0");
        }
        auto mparams = llama_model_default_params();
        mparams.devices = devices;
        mparams.n_gpu_layers = 99;
        mparams.main_gpu = 0;
        mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(nullptr, &llama_model_free);
        const double load_ms = host_time([&] { model.reset(llama_model_load_from_file(o.model.c_str(), mparams)); });
        if (!model) throw std::runtime_error("Model loading failed");
        if (model_meta(model.get(), "general.architecture") != "qwen35moe" || llama_model_has_encoder(model.get())) {
            throw std::runtime_error("This milestone targets decoder-only qwen35moe GGUF models");
        }
        const auto max_context = llama_model_n_ctx_train(model.get());
        if (max_context <= 0 || o.ctx_size > static_cast<uint32_t>(max_context)) {
            throw std::runtime_error("Requested context exceeds the model's declared training window");
        }
        auto cparams = llama_context_default_params();
        cparams.n_ctx = o.ctx_size;
        cparams.n_batch = o.batch_size;
        cparams.n_ubatch = o.ubatch_size;
        cparams.n_threads = static_cast<int32_t>(o.threads);
        cparams.n_threads_batch = static_cast<int32_t>(o.threads);
        cparams.n_seq_max = 1;
        cparams.n_rs_seq = 0;
        cparams.type_k = GGML_TYPE_F16;
        cparams.type_v = GGML_TYPE_F16;
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        cparams.offload_kqv = true;
        cparams.op_offload = true;
        cparams.no_perf = false;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cparams), &llama_free);
        if (!ctx || !llama_get_memory(ctx.get())) throw std::runtime_error("Context/memory initialization failed");
        const uint64_t required_context = uint64_t(o.prefix_tokens) + o.suffix_tokens + o.decode_tokens;
        if (llama_n_ctx(ctx.get()) < required_context || llama_n_ctx(ctx.get()) > static_cast<uint32_t>(max_context)) {
            throw std::runtime_error("Actual context must cover the workload without exceeding the model's training window");
        }
        metadata.row({"model_load_ms", metric(load_ms)});
        metadata.row({"context_actual", std::to_string(llama_n_ctx(ctx.get()))});
        metadata.row({"batch_actual", std::to_string(llama_n_batch(ctx.get()))});
        metadata.row({"ubatch_actual", std::to_string(llama_n_ubatch(ctx.get()))});
        const auto * vocab = llama_model_get_vocab(model.get());
        const uint32_t prompt_count = o.prefix_tokens + o.suffix_tokens;
        const auto a = synthetic_tokens(vocab, prompt_count, o.seed);
        const auto b = synthetic_tokens(vocab, prompt_count, seed_b);
        if (a == b) throw std::runtime_error("Interference sequence B must differ from A");
        const Identity identity = make_identity(o, ctx.get(), model.get(), devices[0], a);
        metadata.row({"canonical_identity", identity.configuration});
        metadata.row({"prefix_processed_tokens", std::to_string(identity.processed_tokens)});
        Evaluator evaluator(ctx.get());

        current.mode = "recompute";
        current.trial = bench::Trial{bench::Phase::validation, 0, 0, bench::Mode::recompute};
        run_interference(ctx.get(), evaluator, b, current);
        const auto baseline = recompute_request(o, ctx.get(), evaluator, vocab, a, current);
        derive_metrics(current, o, bench::Mode::recompute);
        current.parity_ms = 0;
        current.status = "reference";
        current.detail = "Finite logits; unmeasured golden continuation; excluded from statistics";
        records.push_back(current);
        current_recorded = true;
        write_continuation(continuations, current, baseline);
        write_result(results, current, o);

        qwen_cache::MemoryStore memory(o.limits);
        qwen_cache::DirectFileStore files(o.output_dir / "cache", o.limits);
        current = Result{};
        current.mode = "dram";
        current_recorded = false;
        prepare_snapshot(memory, o, ctx.get(), evaluator, identity, a, current);
        const Result dram_setup = current;
        write_setup(metadata, dram_setup);
        current = Result{};
        current.mode = "io-uring-direct";
        prepare_snapshot(files, o, ctx.get(), evaluator, identity, a, current, o.output_dir / "cache");
        const Result file_setup = current;
        write_setup(metadata, file_setup);

        for (const auto & trial : bench::make_schedule(o.repetitions, o.warmup)) {
            current = Result{};
            current.trial = trial;
            current.mode = bench::mode_name(trial.mode);
            current_recorded = false;
            run_interference(ctx.get(), evaluator, b, current);
            Continuation output;
            switch (trial.mode) {
                case bench::Mode::recompute:
                    output = recompute_request(o, ctx.get(), evaluator, vocab, a, current);
                    break;
                case bench::Mode::dram:
                    output = cached_request(memory, o, ctx.get(), evaluator, vocab, identity, a, dram_setup, current);
                    break;
                case bench::Mode::direct_file:
                    output = cached_request(files, o, ctx.get(), evaluator, vocab, identity, a, file_setup, current);
                    break;
            }
            qwen_cache::LogitComparison comparison;
            current.parity_ms = host_time([&] {
                comparison = qwen_cache::compare_logits(baseline.logits, output.logits, 1e-4, 1e-4);
                current.tokens_equal = baseline.tokens == output.tokens;
            });
            current.max_abs_error = comparison.max_abs_error;
            current.max_rel_error = comparison.max_rel_error;
            const bool passed = comparison.passed && current.tokens_equal;
            current.status = passed ? "pass" : "fail";
            current.reused_tokens = passed && trial.mode != bench::Mode::recompute ? o.prefix_tokens : 0;
            current.detail = comparison.detail;
            if (!current.tokens_equal) current.detail += "; greedy continuation mismatch";
            if (trial.mode == bench::Mode::direct_file) current.detail += "; io_uring+O_DIRECT bypasses OS page cache; load includes allocation/copy/CRC, not raw SSD latency";
            if (passed) derive_metrics(current, o, trial.mode);
            records.push_back(current);
            current_recorded = true;
            write_continuation(continuations, current, output);
            write_result(results, current, o);
            std::cout << bench::phase_name(trial.phase) << ' ' << trial.trial << '/' << trial.order
                      << ' ' << current.mode << ": " << current.status << '\n';
            if (!passed) throw std::runtime_error("Trial parity failed; refusing successful aggregate statistics");
        }
        const bool complete = write_summaries(summaries, records, o);
        summary_written = true;
        metadata.row({"model_parity", complete ? "passed_for_this_case_and_recorded_trials_only" : "failed"});
        metadata.row({"benchmark_status", complete ? "complete" : "failed_or_incomplete"});
        metadata.row({"physical_ssd_latency", "not_validated"});
        metadata.row({"trace_replay", "not_run"});
        return complete ? 0 : 1;
    } catch (const std::exception & e) {
        current.status = "error";
        current.detail = e.what();
        current.reused_tokens = 0;
        if (!current_recorded) {
            records.push_back(current);
            write_result(results, current, o);
        }
        metadata.row({"model_parity", "error_or_incomplete"});
        metadata.row({"benchmark_status", "failed_or_incomplete"});
        metadata.row({"error", e.what()});
        if (!summary_written) write_summaries(summaries, records, o, true);
        throw;
    }
}

int main(int argc, char ** argv) {
    // Parse before touching any backend. Build-time --help/validation is model-free.
    if (argc == 1 || (argc == 2 && std::string(argv[1]) == "--help")) {
        usage();
        return 0;
    }
    try {
        const Options options = parse_options(argc, argv);
        return run(options);
    } catch (const std::exception & e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
