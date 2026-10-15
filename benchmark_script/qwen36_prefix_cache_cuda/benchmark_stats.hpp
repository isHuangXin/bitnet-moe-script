#ifndef QWEN36_PREFIX_BENCHMARK_STATS_HPP
#define QWEN36_PREFIX_BENCHMARK_STATS_HPP

// Host-only scheduling and instrumented/token-ready benchmark statistics.
// No model, clock, storage or I/O dependencies.
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen_cache::bench {

enum class Mode { recompute, dram, direct_file };
enum class Phase { validation, warmup, measurement };

inline const char * mode_name(Mode mode) {
    switch (mode) {
        case Mode::recompute: return "recompute";
        case Mode::dram: return "dram";
        case Mode::direct_file: return "io-uring-direct";
    }
    throw std::invalid_argument("invalid benchmark mode");
}

inline const char * phase_name(Phase phase) {
    switch (phase) {
        case Phase::validation: return "validation";
        case Phase::warmup: return "warmup";
        case Phase::measurement: return "measurement";
    }
    throw std::invalid_argument("invalid benchmark phase");
}

struct Trial {
    Phase phase = Phase::validation;
    uint32_t trial = 0;
    uint32_t order = 0;
    Mode mode = Mode::recompute;
};

inline void validate_counts(uint32_t repetitions, uint32_t warmup) {
    if (repetitions < 1 || repetitions > 1000 || warmup < 1 || warmup > 1000) {
        throw std::invalid_argument("repetitions and warmup must be in [1, 1000]");
    }
}

inline std::vector<Trial> make_schedule(uint32_t repetitions, uint32_t warmup) {
    validate_counts(repetitions, warmup);
    constexpr Mode modes[] = {Mode::recompute, Mode::dram, Mode::direct_file};
    std::vector<Trial> schedule;
    schedule.reserve(3 * (repetitions + warmup));
    // Zero-based trial and order; restart the rotation for each phase. The
    // independent golden validation request is deliberately not in this list.
    for (Phase phase : {Phase::warmup, Phase::measurement}) {
        const uint32_t count = phase == Phase::warmup ? warmup : repetitions;
        for (uint32_t round = 0; round < count; ++round) {
            for (uint32_t order = 0; order < 3; ++order) {
                schedule.push_back({phase, round, order, modes[(round + order) % 3]});
            }
        }
    }
    return schedule;
}

struct Timings {
    double prefix_ms;
    double tail_ms;
    double prompt_ready_ms;
    double ttft_ms;
    double last_sample_ms;
    double request_ms;
    double decode_forward_ms;
};

struct Rates {
    double actual_prefill_eval_tps;
    double effective_prompt_tps;
    std::optional<double> conditioned_decode_forward_tps;
    std::optional<double> generation_after_first_tps;
    double end_to_end_output_tps;
};

inline Rates derive_rates(Mode mode, uint32_t P, uint32_t S, uint32_t D, const Timings & t) {
    (void)mode_name(mode); // Reject unknown enum values rather than treating them as cache hits.
    if (P == 0 || S == 0) throw std::invalid_argument("P and S must be positive");
    const auto positive = [](double value) { return std::isfinite(value) && value > 0; };
    if (!positive(t.tail_ms) || !positive(t.prompt_ready_ms) || !positive(t.ttft_ms) ||
        !positive(t.last_sample_ms) || !positive(t.request_ms) ||
        (D == 0 ? t.decode_forward_ms != 0 : !positive(t.decode_forward_ms)) ||
        !std::isfinite(t.prefix_ms) || (mode == Mode::recompute ? t.prefix_ms <= 0 : t.prefix_ms != 0)) {
        throw std::invalid_argument("timings must be positive and finite; cache prefix_ms and D=0 forward time must be zero");
    }
    const long double prefill_ms = static_cast<long double>(t.prefix_ms) + t.tail_ms;
    if (prefill_ms > t.prompt_ready_ms || t.prompt_ready_ms > t.ttft_ms || t.ttft_ms > t.last_sample_ms ||
        (D == 0 ? (t.last_sample_ms != t.ttft_ms || t.request_ms < t.last_sample_ms)
                : (t.last_sample_ms >= t.request_ms || t.decode_forward_ms > t.request_ms - t.ttft_ms ||
                   (D == 1 ? t.last_sample_ms != t.ttft_ms : t.last_sample_ms <= t.ttft_ms)))) {
        throw std::invalid_argument("inconsistent prompt/sample/request timing boundaries");
    }
    // Widen before adding: the helper must never wrap uint32_t P+S. The probe
    // separately enforces P+S+D against its actual context/model budget.
    const uint64_t prompt_tokens = static_cast<uint64_t>(P) + S;
    const auto rate = [](uint64_t count, long double elapsed_ms) {
        const double value = static_cast<double>(1000.0L * count / elapsed_ms);
        if (!std::isfinite(value) || value <= 0) throw std::invalid_argument("invalid or nonfinite derived token rate");
        return value;
    };
    Rates result{
        mode == Mode::recompute ? rate(prompt_tokens, prefill_ms) : rate(S, t.tail_ms),
        rate(prompt_tokens, t.prompt_ready_ms),
        std::nullopt,
        std::nullopt,
        rate(D == 0 ? 1 : D, t.last_sample_ms)
    };
    // D=0 samples one first token from prefill logits without forwarding it.
    // For D>0, D forward calls include the update after the last sampled token.
    // Neither the forward-only denominator nor Tend is the last-output time.
    if (D > 0) result.conditioned_decode_forward_tps = rate(D, t.decode_forward_ms);
    if (D > 1) result.generation_after_first_tps = rate(D - 1, t.last_sample_ms - t.ttft_ms);
    return result;
}

struct Observation {
    Trial trial;
    bool passed = false;
    std::optional<double> value;
};

struct Summary {
    uint32_t n_expected = 0;
    uint32_t n_observed = 0;
    std::string status;
    std::optional<double> mean;
    std::optional<double> stddev;
};

inline Summary summarize(Mode mode, uint32_t repetitions, const std::vector<Observation> & observations,
                         bool allow_all_na = false, bool force_failed = false) {
    (void)mode_name(mode);
    validate_counts(repetitions, 1);
    Summary result;
    result.n_expected = repetitions;
    std::set<uint32_t> attempted;
    bool failed = force_failed;
    uint32_t numeric_count = 0;
    uint32_t null_count = 0;
    long double mean = 0;
    long double squared_deviations = 0;
    for (const Observation & observation : observations) {
        const Trial & trial = observation.trial;
        if (trial.phase != Phase::measurement || trial.mode != mode) continue;
        const bool unique = attempted.insert(trial.trial).second;
        // Count unique attempted IDs even when they are outside the expected
        // range. Do not turn repeated/malformed/failed rows into successful data.
        if (!unique || trial.trial >= repetitions || trial.order >= 3 ||
            (trial.trial % 3 + trial.order) % 3 != static_cast<uint32_t>(mode) || !observation.passed) {
            failed = true;
        }
        if (!observation.value) {
            ++null_count;
            if (!allow_all_na) failed = true;
        } else if (!std::isfinite(*observation.value)) {
            failed = true;
        } else {
            ++numeric_count;
            // Welford in wider precision avoids overflowing a double sum or
            // sum-of-squares for otherwise finite, representable statistics.
            const long double delta = static_cast<long double>(*observation.value) - mean;
            mean += delta / numeric_count;
            squared_deviations += delta * (static_cast<long double>(*observation.value) - mean);
        }
    }
    if (attempted.size() > std::numeric_limits<uint32_t>::max()) throw std::length_error("too many observed trial IDs");
    result.n_observed = static_cast<uint32_t>(attempted.size());
    if (null_count != 0 && numeric_count != 0) failed = true;
    // Status precedence: any explicit/structural failure beats missing data.
    // Incomplete is reserved for a valid but unfinished measurement set.
    // Only pass has numerical statistics; complete all-NA has not_applicable.
    if (failed) {
        result.status = "failed";
    } else if (result.n_observed != repetitions) {
        result.status = "incomplete";
    } else if (allow_all_na && null_count == repetitions) {
        result.status = "not_applicable";
    } else {
        const double average = static_cast<double>(mean);
        const double deviation = numeric_count > 1
            ? static_cast<double>(std::sqrt(squared_deviations / (numeric_count - 1))) : 0;
        if (numeric_count != repetitions || !std::isfinite(average) || !std::isfinite(deviation)) {
            result.status = "failed";
        } else {
            result.status = "pass";
            result.mean = average;
            result.stddev = deviation; // Singleton stddev=0, matching llama-bench.
        }
    }
    return result;
}

} // namespace qwen_cache::bench
#endif
