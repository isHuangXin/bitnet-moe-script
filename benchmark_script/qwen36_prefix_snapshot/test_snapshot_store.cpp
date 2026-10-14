#include "snapshot_store.hpp"
#include "benchmark_stats.hpp"

#include <csignal>
#include <iostream>
#include <sys/resource.h>
#include <sys/wait.h>
#include <type_traits>

namespace fs = std::filesystem;
using namespace qwen_cache;

namespace {

const fs::path artifact_root = "/data2/huangxin/qwen36_cache_experiments";
size_t checks = 0;

void require(bool condition, const std::string & message) {
    ++checks;
    if (!condition) throw std::runtime_error("test failed: " + message);
}

template <typename Function>
void throws(Function && function, const std::string & message) {
    bool caught = false;
    try { function(); } catch (const std::exception &) { caught = true; }
    require(caught, message);
}

Limits small_limits() { return {8ULL << 20, 32ULL << 20, 0}; }

void require_artifact_path(const fs::path & path) {
    const fs::path absolute = fs::absolute(path).lexically_normal();
    auto current = absolute.begin();
    for (const auto & part : artifact_root) {
        if (current == absolute.end() || *current != part) {
            throw std::invalid_argument("test artifacts must stay under " + artifact_root.string());
        }
        ++current;
    }
}

struct RunDirectory {
    fs::path path;
    explicit RunDirectory(const fs::path & parent) {
        require_artifact_path(parent);
        // A supplied fixture parent may be 0755: the unique mkdtemp child is
        // private. Reject symlink components, foreign ownership and writable
        // shared parents without chmod'ing or deleting any existing directory.
        fs::path checked;
        struct stat st{};
        for (const auto & component : fs::absolute(parent)) {
            if (component == "..") throw std::invalid_argument("parent traversal in test fixture root");
            checked /= component;
            if (::lstat(checked.c_str(), &st) != 0) {
                if (errno != ENOENT) throw detail::io_error("inspect test fixture parent");
                if (::mkdir(checked.c_str(), 0700) != 0 && errno != EEXIST) throw detail::io_error("create test fixture parent");
                if (::lstat(checked.c_str(), &st) != 0) throw detail::io_error("stat test fixture parent");
            }
            if (!S_ISDIR(st.st_mode)) throw std::runtime_error("test fixture parent contains a symlink or nondirectory");
        }
        if (st.st_uid != ::geteuid() || (st.st_mode & 0022) != 0) {
            throw std::runtime_error("test fixture parent must be locally owned and not writable by group/others");
        }
        std::string pattern = (fs::absolute(parent) / "snapshot-store-XXXXXX").string();
        std::vector<char> name(pattern.begin(), pattern.end());
        name.push_back('\0');
        char * created = ::mkdtemp(name.data());
        if (!created) throw detail::io_error("create exclusive test run directory");
        path = created;
    }
    ~RunDirectory() {
        // mkdtemp exclusively created this directory. Never remove the supplied
        // parent or any preexisting entry; remove_all does not follow symlinks.
        std::error_code error;
        fs::remove_all(path, error);
        if (error) std::cerr << "fixture cleanup failed: " << error.message() << '\n';
    }
};

Snapshot sample(size_t state_bytes = 513) {
    Snapshot snapshot;
    const char configuration[] = "fake-config\0model=none;build=test;backend=host";
    snapshot.identity.configuration.assign(configuration, sizeof(configuration) - 1);
    snapshot.identity.prefix_tokens = {0, 17, -1, INT32_MIN, INT32_MAX, 0};
    snapshot.identity.processed_tokens = snapshot.identity.prefix_tokens.size();
    snapshot.state.resize(state_bytes);
    for (size_t i = 0; i < state_bytes; ++i) snapshot.state[i] = static_cast<uint8_t>((i * 131 + 7) & 255);
    if (state_bytes > 2) { snapshot.state[0] = 0; snapshot.state[1] = 255; }
    return snapshot;
}

uint64_t encoded_size(const Snapshot & snapshot) {
    return 64 + snapshot.identity.configuration.size() + snapshot.identity.prefix_tokens.size() * 4 + snapshot.state.size();
}

void write_fixture(const fs::path & path, const std::vector<uint8_t> & bytes) {
    detail::Fd fd(::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600));
    if (fd.get() < 0) throw detail::io_error("create exclusive test fixture");
    detail::write_all(fd.get(), bytes.data(), bytes.size());
}

std::vector<uint8_t> read_fixture(const fs::path & path) {
    detail::Fd fd(::open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC));
    if (fd.get() < 0) throw detail::io_error("read test fixture");
    struct stat st{};
    if (::fstat(fd.get(), &st) != 0) throw detail::io_error("stat test fixture");
    require(st.st_size >= 0 && st.st_size <= (8 << 20), "fixture reads stay bounded");
    std::vector<uint8_t> result(static_cast<size_t>(st.st_size));
    detail::read_all(fd.get(), result.data(), result.size());
    return result;
}

void set_u64(std::vector<uint8_t> & bytes, size_t offset, uint64_t value) {
    require(offset + 8 <= bytes.size(), "patch offset is in bounds");
    for (size_t i = 0; i < 8; ++i) bytes[offset + i] = static_cast<uint8_t>(value >> (8 * i));
}

// Independent, deliberately slow reference CRC for small test fixtures only.
uint64_t reference_crc(const uint8_t * data, size_t size, uint64_t value = 0) {
    for (size_t i = 0; i < size; ++i) {
        value ^= static_cast<uint64_t>(data[i]) << 56;
        for (int bit = 0; bit < 8; ++bit) {
            value = (value & (1ULL << 63)) ? (value << 1) ^ 0x42f0e1eba9ea3693ULL : value << 1;
        }
    }
    return value;
}

void repair_crc(std::vector<uint8_t> & bytes) {
    uint64_t crc = reference_crc(bytes.data(), 56);
    crc = reference_crc(bytes.data() + 64, bytes.size() - 64, crc);
    set_u64(bytes, 56, crc);
}

void require_status(const LookupResult & result, LookupStatus status, const std::string & message) {
    require(result.status == status, message + ": " + result.detail);
    require(static_cast<bool>(result.snapshot) == (status == LookupStatus::hit), message + " pointer/status agreement");
}

template <typename Store>
void common_store_checks(Store & store) {
    const Snapshot first = sample();
    require_status(store.get("missing", first.identity), LookupStatus::miss, "absent key misses");
    store.put("first", first);
    auto hit = store.get("first", first.identity);
    require_status(hit, LookupStatus::hit, "first key roundtrip");
    require(hit.snapshot->state == first.state, "binary state including NUL/255 roundtrips");
    require(hit.snapshot->identity == first.identity, "binary configuration and signed tokens roundtrip");
    Snapshot second = sample(4097);
    second.identity.configuration += ";different-model";
    second.identity.prefix_tokens[0] = 1;
    second.state[2] ^= 255;
    store.put("second-Key_2", second);
    auto other = store.get("second-Key_2", second.identity);
    require_status(other, LookupStatus::hit, "second independent key roundtrip");
    require(other.snapshot->state == second.state, "multiple keys retain distinct state");
    throws([&] { store.put("first", first); }, "identical duplicate is rejected");
    throws([&] { store.put("first", second); }, "conflicting duplicate is rejected");
    require(store.get("first", first.identity).snapshot->state == first.state, "duplicate cannot replace published state");
    Identity different = first.identity;
    different.configuration += ";different-library";
    require_status(store.get("first", different), LookupStatus::incompatible, "configuration mismatch");
    different = first.identity;
    different.prefix_tokens[1] += 1;
    require_status(store.get("first", different), LookupStatus::incompatible, "exact token mismatch");
    different = first.identity;
    different.processed_tokens -= 1;
    require_status(store.get("first", different), LookupStatus::incompatible, "processed boundary mismatch");
    different = first.identity;
    different.prefix_tokens.pop_back();
    different.processed_tokens = different.prefix_tokens.size();
    require_status(store.get("first", different), LookupStatus::incompatible, "shorter prefix mismatch");
    Snapshot invalid = first;
    invalid.identity.processed_tokens -= 1;
    throws([&] { store.put("bad-boundary", invalid); }, "put refuses wrong processed boundary");
    invalid = first;
    invalid.state.clear();
    throws([&] { store.put("empty-state", invalid); }, "put refuses empty state");
    invalid = first;
    invalid.identity.configuration.clear();
    throws([&] { store.put("empty-identity", invalid); }, "put refuses empty configuration");
    invalid = first;
    invalid.identity.configuration.assign((1U << 20) + 1, 'x');
    throws([&] { store.put("large-identity", invalid); }, "put bounds configuration size");
    invalid = first;
    invalid.identity.prefix_tokens.resize((1U << 20) + 1);
    invalid.identity.processed_tokens = invalid.identity.prefix_tokens.size();
    throws([&] { store.put("large-prefix", invalid); }, "put bounds token count");
    const std::vector<std::string> unsafe = {"", "../escape", "a/b", ".", "a.b", "a\nb", std::string("a\0b", 3), std::string(129, 'a'), "\xff"};
    for (const auto & key : unsafe) {
        throws([&] { store.put(key, first); }, "unsafe put key rejected");
        throws([&] { (void)store.get(key, first.identity); }, "unsafe lookup key rejected");
    }
    Snapshot empty_prefix = first;
    empty_prefix.identity.prefix_tokens.clear();
    empty_prefix.identity.processed_tokens = 0;
    store.put("empty-prefix", empty_prefix);
    require_status(store.get("empty-prefix", empty_prefix.identity), LookupStatus::hit, "zero-token exact boundary is supported");
}

void test_memory() {
    static_assert(std::is_same<decltype(LookupResult{}.snapshot), std::shared_ptr<const Snapshot>>::value,
                  "lookup state must be immutable");
    MemoryStore memory(small_limits());
    common_store_checks(memory);
    const auto identity = sample().identity;
    const auto first = memory.get("first", identity);
    const auto second = memory.get("first", identity);
    require(first.snapshot.get() == second.snapshot.get(), "DRAM lookups share the same immutable snapshot");
    require(first.snapshot->state.data() == second.snapshot->state.data(), "DRAM lookup does not copy state bytes");
    std::shared_ptr<const Snapshot> retained;
    {
        MemoryStore temporary(small_limits());
        temporary.put("retained", sample());
        retained = temporary.get("retained", identity).snapshot;
    }
    require(retained && retained->state == sample().state, "shared hit outlives store safely");
    throws([] { MemoryStore invalid({0, 100, 0}); }, "zero blob limit rejected");
    throws([] { MemoryStore invalid({100, 0, 0}); }, "zero store limit rejected");
}

void test_files(const fs::path & run) {
    const fs::path root = run / "roundtrip";
    BufferedFileStore files(root, small_limits());
    common_store_checks(files);
    struct stat st{};
    require(::stat(root.c_str(), &st) == 0 && (st.st_mode & 0777) == 0700, "new store root has mode 0700");
    require(::stat((root / "first.qcache").c_str(), &st) == 0 && (st.st_mode & 0777) == 0600,
            "published snapshot has mode 0600");
    require(st.st_nlink == 1, "publication leaves exactly one link");
    const auto bytes = read_fixture(root / "first.qcache");
    require(bytes.size() == encoded_size(sample()), "serialized quota includes all metadata");
    const std::vector<uint8_t> expected_magic = {'Q', 'W', 'N', 'S', 'N', 'A', 'P', 0};
    require(std::equal(expected_magic.begin(), expected_magic.end(), bytes.begin()), "versioned magic");
    require(bytes[8] == 1 && bytes[9] == 0 && bytes[10] == 0 && bytes[11] == 0 && bytes[12] == 64,
            "canonical LE version/header size");
    const size_t tokens = 64 + sample().identity.configuration.size();
    require(bytes[tokens + 4] == 17 && bytes[tokens + 5] == 0, "positive token canonical LE");
    require(bytes[tokens + 8] == 255 && bytes[tokens + 11] == 255, "negative token canonical LE");
    require(bytes[tokens + 12] == 0 && bytes[tokens + 15] == 128, "INT32_MIN canonical LE");
    auto patched = bytes;
    repair_crc(patched);
    require(patched == bytes, "published checksum matches independent ECMA reference");
    for (const auto & entry : fs::directory_iterator(root)) {
        require(entry.path().extension() == ".qcache", "successful publication leaves no staging name");
    }
    Snapshot large = sample((2U << 20) + 17);
    large.identity.prefix_tokens.resize(5000);
    for (size_t i = 0; i < large.identity.prefix_tokens.size(); ++i) {
        large.identity.prefix_tokens[i] = static_cast<int32_t>(i) - 2500;
    }
    large.identity.processed_tokens = large.identity.prefix_tokens.size();
    files.put("chunked", large);
    const auto loaded = files.get("chunked", large.identity);
    require_status(loaded, LookupStatus::hit, "chunked record roundtrip");
    require(loaded.snapshot->state == large.state, "multi-megabyte payload roundtrip");
    Identity mismatch = large.identity;
    mismatch.configuration += "-mismatch";
    require_status(files.get("chunked", mismatch), LookupStatus::incompatible, "chunked checksum on incompatible record");
    BufferedFileStore fresh(root, small_limits());
    require(fresh.get("first", sample().identity).snapshot->state == sample().state, "fresh store object reads persisted state");
    require(fresh.get("first", sample().identity).detail.find("buffered-file") != std::string::npos,
            "reader labels buffered-file rather than physical SSD");
}

void test_corruption(const fs::path & run) {
    const fs::path root = run / "corruption";
    BufferedFileStore files(root, small_limits());
    const Snapshot snapshot = sample(37);
    files.put("baseline", snapshot);
    const auto original = read_fixture(root / "baseline.qcache");
    auto bad = [&](const std::string & key, std::vector<uint8_t> bytes) {
        write_fixture(root / (key + ".qcache"), bytes);
        require_status(files.get(key, snapshot.identity), LookupStatus::corrupt, "reject " + key);
    };
    auto bytes = original;
    bytes.back() ^= 1;
    bad("payload-checksum", bytes);
    bytes = original;
    bytes[64] ^= 1;
    bad("metadata-checksum", bytes);
    bytes = original;
    bytes[64 + snapshot.identity.configuration.size()] ^= 1;
    bad("token-checksum", bytes);
    bytes = original;
    bytes[56] ^= 1;
    bad("checksum-field", bytes);
    bytes = original;
    bytes[0] ^= 1;
    bad("magic", bytes);
    bytes = original;
    bytes[8] = 2;
    repair_crc(bytes);
    bad("unknown-version", bytes);
    bytes = original;
    bytes[12] = 63;
    bad("header-size", bytes);
    for (const auto & field : std::vector<std::pair<std::string, size_t>>{
             {"configuration-length", 16}, {"token-count", 24}, {"state-length", 40}, {"record-length", 48}}) {
        bytes = original;
        set_u64(bytes, field.second, UINT64_MAX);
        repair_crc(bytes);
        bad("oversized-" + field.first, bytes);
    }
    bytes = original;
    set_u64(bytes, 32, snapshot.identity.processed_tokens + 1);
    repair_crc(bytes);
    bad("wrong-processed-boundary", bytes);
    bytes = original;
    set_u64(bytes, 40, 0);
    repair_crc(bytes);
    bad("empty-state", bytes);
    bytes = original;
    set_u64(bytes, 16, 0);
    repair_crc(bytes);
    bad("empty-configuration", bytes);
    for (size_t size : {size_t(0), size_t(1), size_t(63), size_t(64), original.size() - 1}) {
        bytes = original;
        bytes.resize(size);
        bad("truncated-" + std::to_string(size), bytes);
    }
    bytes = original;
    bytes.push_back(42);
    bad("trailing-byte", bytes);
    bytes = original;
    set_u64(bytes, 48, original.size() - 1);
    repair_crc(bytes);
    bad("inconsistent-record-length", bytes);
    bytes = original;
    bytes[64] ^= 1;
    repair_crc(bytes);
    write_fixture(root / "valid-wrong-identity.qcache", bytes);
    require_status(files.get("valid-wrong-identity", snapshot.identity), LookupStatus::incompatible,
                   "identity checked independently of CRC");
    auto lower = small_limits();
    lower.max_blob_bytes = original.size() - 1;
    BufferedFileStore bounded(root, lower);
    require_status(bounded.get("baseline", snapshot.identity), LookupStatus::corrupt, "reader applies blob bound");
    lower = small_limits();
    lower.max_store_bytes = original.size() - 1;
    BufferedFileStore quota_bounded(root, lower);
    require_status(quota_bounded.get("baseline", snapshot.identity), LookupStatus::corrupt, "reader applies entry store bound");
}

void test_quota_and_staging(const fs::path & run) {
    const Snapshot snapshot = sample(31);
    const uint64_t size = encoded_size(snapshot);
    Limits limits{size, size * 2, 0};
    MemoryStore memory(limits);
    memory.put("one", snapshot);
    memory.put("two", snapshot);
    throws([&] { memory.put("three", snapshot); }, "DRAM total quota exact boundary");
    Snapshot larger = snapshot;
    larger.state.push_back(0);
    throws([&] { memory.put("larger", larger); }, "blob quota includes metadata");
    require_status(memory.get("one", snapshot.identity), LookupStatus::hit, "quota failure retains DRAM state");
    const fs::path root = run / "quota";
    BufferedFileStore files(root, limits);
    files.put("one", snapshot);
    const fs::path unfinished = root / ".qcache-tmp-interrupted";
    write_fixture(unfinished, {9});
    require_status(files.get("interrupted", snapshot.identity), LookupStatus::miss, "unfinished temporary is invisible");
    throws([&] { files.put("two", snapshot); }, "unfinished staging bytes count against file quota");
    require(read_fixture(unfinished) == std::vector<uint8_t>{9}, "quota failure preserves preexisting staging fixture");
    require(std::distance(fs::directory_iterator(root), fs::directory_iterator{}) == 2, "no new staging on quota failure");
    require(fs::remove(unfinished), "remove only this test's own staging fixture");
    files.put("two", snapshot);
    throws([&] { files.put("three", snapshot); }, "published file quota enforced");
    throws([&] { files.put("larger", larger); }, "file blob quota includes metadata");
    require_status(files.get("one", snapshot.identity), LookupStatus::hit, "quota failure retains published state");
    BufferedFileStore fresh(root, limits);
    throws([&] { fresh.put("three", snapshot); }, "fresh writer accounts for published bytes");
    const auto before = read_fixture(root / "one.qcache");
    Snapshot conflict = snapshot;
    conflict.state[0] ^= 1;
    throws([&] { fresh.put("one", conflict); }, "fresh writer cannot overwrite publication");
    require(read_fixture(root / "one.qcache") == before, "conflict leaves bytes unchanged");
    Limits reserved = small_limits();
    reserved.disk_reserve_bytes = UINT64_MAX;
    const fs::path reserve_root = run / "reserve";
    BufferedFileStore reserve(reserve_root, reserved);
    throws([&] { reserve.put("blocked", snapshot); }, "statvfs disk reserve enforced without overflow");
    require(fs::is_empty(reserve_root), "reserve failure creates no files");
    const fs::path incomplete_root = run / "incomplete";
    BufferedFileStore incomplete(incomplete_root, small_limits());
    auto partial = before;
    partial.resize(partial.size() / 2);
    write_fixture(incomplete_root / ".qcache-tmp-unfinished", partial);
    require_status(incomplete.get("unfinished", snapshot.identity), LookupStatus::miss, "partial unpublished envelope invisible");
    incomplete.put("unfinished", snapshot);
    require_status(incomplete.get("unfinished", snapshot.identity), LookupStatus::hit, "publication ignores unfinished unrelated name");
    require(read_fixture(incomplete_root / ".qcache-tmp-unfinished") == partial, "unrelated unfinished file never cleaned");
}

void test_atomic_failure_cleanup(const fs::path & run) {
    const fs::path root = run / "publication";
    BufferedFileStore files(root, small_limits());
    const Snapshot snapshot = sample(31);
    files.put("existing", snapshot);
    const auto original = read_fixture(root / "existing.qcache");
    const fs::path unrelated = root / ".qcache-tmp-preserve";
    write_fixture(unrelated, {7, 8, 9});
    auto directory = detail::open_root(root);
    fs::path staging;
    {
        // Exercise the actual no-replace syscall, bypassing put's early
        // duplicate check to simulate a destination appearing before publish.
        detail::TempFile temporary(directory.get());
        staging = root / temporary.name();
        detail::write_record(temporary.fd(), snapshot, encoded_size(snapshot));
        require_status(files.get("pending", snapshot.identity), LookupStatus::miss, "fsynced staging remains invisible");
        throws([&] { temporary.publish("existing.qcache"); }, "atomic publication rejects late destination collision");
        require(read_fixture(root / "existing.qcache") == original, "late collision never overwrites original bytes");
    }
    require(!fs::exists(staging), "failed publication cleans its own temporary file");
    {
        detail::TempFile temporary(directory.get());
        staging = root / temporary.name();
        detail::write_record(temporary.fd(), snapshot, encoded_size(snapshot));
        temporary.publish("pending.qcache");
        require(!fs::exists(staging), "publication atomically removes temporary name");
        require_status(files.get("pending", snapshot.identity), LookupStatus::hit, "publication exposes only a complete one-link record");
        require(::fsync(directory.get()) == 0, "sync owned publication test directory");
    }
    // Real bounded I/O failure, isolated to a child. No filesystem is filled and
    // the parent process's signal disposition/resource limits remain unchanged.
    std::cout.flush();
    const pid_t child = ::fork();
    if (child < 0) throw detail::io_error("fork bounded write-failure test");
    if (child == 0) {
        struct rlimit limit{};
        if (::getrlimit(RLIMIT_FSIZE, &limit) != 0 || std::signal(SIGXFSZ, SIG_IGN) == SIG_ERR) ::_exit(90);
        limit.rlim_cur = 96;
        if (::setrlimit(RLIMIT_FSIZE, &limit) != 0) ::_exit(91);
        try {
            files.put("io-failed", sample(8193));
        } catch (const std::exception & error) {
            ::_exit(std::string(error.what()).find("write buffered-file snapshot") != std::string::npos ? 0 : 92);
        }
        ::_exit(93);
    }
    int status = 0;
    pid_t waited;
    do { waited = ::waitpid(child, &status, 0); } while (waited < 0 && errno == EINTR);
    require(waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 0, "injected EFBIG write failure handled");
    require_status(files.get("io-failed", snapshot.identity), LookupStatus::miss, "failed write never publishes partial state");
    require(std::distance(fs::directory_iterator(root), fs::directory_iterator{}) == 3, "write failure removes only newly owned staging");
    require(read_fixture(unrelated) == std::vector<uint8_t>({7, 8, 9}), "failure cleanup preserves preexisting staging bytes");
    require(read_fixture(root / "existing.qcache") == original, "I/O failure preserves published bytes");
    files.put("after-failure", snapshot);
    require_status(files.get("after-failure", snapshot.identity), LookupStatus::hit, "store remains usable after I/O failure");
}

void test_path_safety(const fs::path & run) {
    const fs::path root = run / "safety";
    BufferedFileStore files(root, small_limits());
    const auto snapshot = sample();
    files.put("valid", snapshot);
    const fs::path external = run / "external-owned-fixture";
    write_fixture(external, {1, 2, 3});
    require(::symlink(external.c_str(), (root / "symlink.qcache").c_str()) == 0, "create owned symlink fixture");
    require_status(files.get("symlink", snapshot.identity), LookupStatus::corrupt, "O_NOFOLLOW rejects snapshot symlink");
    throws([&] { files.put("symlink", snapshot); }, "symlink cannot be overwritten");
    throws([&] { files.put("other", snapshot); }, "quota scan rejects unsafe directory entries");
    require(read_fixture(external) == std::vector<uint8_t>({1, 2, 3}), "symlink target remains untouched");
    require(fs::remove(root / "symlink.qcache"), "remove own symlink only");
    require(::link((root / "valid.qcache").c_str(), (root / "hardlink.qcache").c_str()) == 0, "create owned hardlink fixture");
    require_status(files.get("hardlink", snapshot.identity), LookupStatus::corrupt, "multiply linked external snapshots rejected");
    require(fs::remove(root / "hardlink.qcache"), "remove own hardlink only");
    require(::chmod((root / "valid.qcache").c_str(), 0644) == 0, "change own fixture mode");
    require_status(files.get("valid", snapshot.identity), LookupStatus::corrupt, "nonprivate snapshot rejected");
    require(::chmod((root / "valid.qcache").c_str(), 0600) == 0, "restore own fixture mode");
    require(::mkfifo((root / "fifo.qcache").c_str(), 0600) == 0, "create owned FIFO fixture");
    require_status(files.get("fifo", snapshot.identity), LookupStatus::corrupt, "FIFO rejected without blocking");
    require(fs::remove(root / "fifo.qcache"), "remove own FIFO only");
    require(::mkdir((root / "directory.qcache").c_str(), 0700) == 0, "create owned directory fixture");
    require_status(files.get("directory", snapshot.identity), LookupStatus::corrupt, "directory cannot be a snapshot");
    const fs::path public_root = run / "public";
    require(::mkdir(public_root.c_str(), 0755) == 0, "create own nonprivate root fixture");
    require(::chmod(public_root.c_str(), 0755) == 0, "set exact nonprivate fixture mode despite umask");
    throws([&] { BufferedFileStore invalid(public_root, small_limits()); }, "existing nonprivate root rejected");
    struct stat st{};
    require(::stat(public_root.c_str(), &st) == 0 && (st.st_mode & 0777) == 0755, "existing root is never chmod'ed");
    const fs::path linked_root = run / "linked-root";
    require(::symlink(root.c_str(), linked_root.c_str()) == 0, "create own linked root fixture");
    throws([&] { BufferedFileStore invalid(linked_root, small_limits()); }, "symlink root rejected");
    throws([&] { BufferedFileStore invalid(linked_root / "child", small_limits()); }, "symlink ancestor rejected");
    throws([&] { BufferedFileStore invalid(root / ".." / "other", small_limits()); }, "parent traversal rejected");
    throws([&] { BufferedFileStore invalid(external / "child", small_limits()); }, "nondirectory parent rejected");
    throws([&] { BufferedFileStore invalid("", small_limits()); }, "empty root rejected");
    require_status(files.get("valid", snapshot.identity), LookupStatus::hit, "valid entry survives safety tests");
}

int child_read(const fs::path & root) {
    require_artifact_path(root);
    BufferedFileStore reader(root, small_limits());
    const auto expected = sample(8193);
    const auto result = reader.get("persisted", expected.identity);
    require_status(result, LookupStatus::hit, "fresh exec reader hit");
    require(result.snapshot->state == expected.state, "fresh exec reader verifies every byte");
    std::cout << "[PASS] separate-process fresh reader (fork/exec, no model)\n";
    return 0;
}

void test_fresh_process(const fs::path & run) {
    const fs::path root = run / "fresh-process";
    {
        BufferedFileStore writer(root, small_limits());
        writer.put("persisted", sample(8193));
    }
    std::cout.flush();
    const pid_t child = ::fork();
    if (child < 0) throw detail::io_error("fork fresh reader");
    if (child == 0) {
        ::execl("/proc/self/exe", "test_snapshot_store", "--child-read", root.c_str(), static_cast<char *>(nullptr));
        ::_exit(127);
    }
    int status = 0;
    pid_t waited;
    do { waited = ::waitpid(child, &status, 0); } while (waited < 0 && errno == EINTR);
    require(waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 0, "fresh exec reader exits successfully");
}

struct FakeModel {
    std::vector<int32_t> history;
    uint64_t recurrent = 0;
    uint64_t convolution = 0;

    void clear() { history.clear(); recurrent = 0; convolution = 0; }
    void step(int32_t token, size_t position) {
        if (position != history.size()) throw std::runtime_error("fake suffix position mismatch");
        recurrent = recurrent * 6364136223846793005ULL + static_cast<uint32_t>(token) + position + 1;
        convolution = convolution * 31 + static_cast<uint32_t>(token);
        history.push_back(token);
    }
    void prefill(const std::vector<int32_t> & tokens) {
        for (int32_t token : tokens) step(token, history.size());
    }
    std::vector<float> logits() const {
        if (history.empty()) throw std::runtime_error("fake logits require suffix processing");
        uint64_t attention = 0;
        for (size_t i = 0; i < history.size(); ++i) attention += static_cast<uint32_t>(history[i]) * (i + 1);
        std::vector<float> result(13);
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] = static_cast<float>(((recurrent >> (i * 3)) ^ (convolution + i * 7) ^ attention) % 10007) / 1024;
        }
        return result;
    }
    std::vector<uint8_t> bytes() const {
        std::vector<uint8_t> result(24 + history.size() * 4);
        detail::put_le(result.data(), history.size(), 8);
        detail::put_le(result.data() + 8, recurrent, 8);
        detail::put_le(result.data() + 16, convolution, 8);
        for (size_t i = 0; i < history.size(); ++i) detail::put_le(result.data() + 24 + i * 4, static_cast<uint32_t>(history[i]), 4);
        return result;
    }
    size_t restore(const uint8_t * bytes, size_t size) {
        require(history.empty() && recurrent == 0 && convolution == 0, "fake importer observes cleared state");
        if (size < 24) throw std::runtime_error("truncated fake state");
        const uint64_t count = detail::get_le(bytes, 8);
        if (count > 100 || size != 24 + count * 4) throw std::runtime_error("invalid fake state");
        recurrent = detail::get_le(bytes + 8, 8);
        convolution = detail::get_le(bytes + 16, 8);
        for (size_t i = 0; i < count; ++i) history.push_back(static_cast<int32_t>(detail::get_le(bytes + 24 + i * 4, 4)));
        return size;
    }
};

std::vector<int32_t> continuation(FakeModel & model, size_t count) {
    std::vector<int32_t> result;
    for (size_t i = 0; i < count; ++i) {
        const auto logits = model.logits();
        const int32_t token = static_cast<int32_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
        result.push_back(token);
        model.step(token, model.history.size());
    }
    return result;
}

template <typename Store>
void fake_aba(Store & store) {
    const std::vector<int32_t> prefix_a = {3, 7, 11, 5};
    const std::vector<int32_t> prefix_b = {9, 1, 8, 2};
    const std::vector<int32_t> suffix = {4, 6};
    FakeModel reference;
    reference.prefill(prefix_a);
    reference.prefill(suffix);
    const auto expected_logits = reference.logits();
    const auto expected_tokens = continuation(reference, 7);
    FakeModel model;
    model.prefill(prefix_a);
    const auto exported = model.bytes();
    auto state = checked_export(exported.size(), [&](uint8_t * destination, size_t size) {
        require(size == exported.size(), "export callback receives exact buffer capacity");
        std::copy(exported.begin(), exported.end(), destination);
        return size;
    }, small_limits());
    const Identity identity{"fake-complete-state;model=none;backend=host", prefix_a, prefix_a.size()};
    store.put("A", {identity, state});
    model.clear();
    model.prefill(prefix_b);
    require(model.bytes() != state, "B actually replaces active A state");
    const auto saved = store.get("A", identity);
    require_status(saved, LookupStatus::hit, "A/B/A looks up saved A");
    std::vector<std::string> order;
    checked_restore(saved.snapshot->state, [&] { order.push_back("clear"); model.clear(); },
                    [&](const uint8_t * bytes, size_t size) { order.push_back("import"); return model.restore(bytes, size); });
    require(order == std::vector<std::string>({"clear", "import"}), "clear happens before successful import");
    require(model.history == prefix_a && model.bytes() == state, "complete recurrent/convolution/history state restored");
    throws([&] { model.step(suffix[0], prefix_a.size() - 1); }, "suffix cannot replay at wrong position");
    model.prefill(suffix);
    const auto compared = compare_logits(expected_logits, model.logits(), 0, 0);
    require(compared.passed, "fake suffix recomputes identical logits");
    require(same_tokens(expected_tokens, continuation(model, 7)), "fake greedy continuation matches after restore");
}

void test_state_protocol(const fs::path & run) {
    MemoryStore memory(small_limits());
    fake_aba(memory);
    BufferedFileStore files(run / "fake-model", small_limits());
    fake_aba(files);
    const Limits limits{32, 64, 0};
    size_t export_calls = 0;
    const auto exporter = [&](uint8_t *, size_t size) { ++export_calls; return size; };
    throws([&] { (void)checked_export(0, exporter, limits); }, "empty export rejected");
    throws([&] { (void)checked_export(33, exporter, limits); }, "oversized export rejected");
    throws([&] { (void)checked_export(SIZE_MAX, exporter, limits); }, "SIZE_MAX export rejected before allocation");
    require(export_calls == 0, "invalid export sizes never invoke callback");
    throws([&] { (void)checked_export(8, [](uint8_t *, size_t n) { return n - 1; }, limits); }, "short export rejected");
    throws([&] { (void)checked_export(8, [](uint8_t *, size_t n) { return n + 1; }, limits); }, "oversized returned export count rejected");
    throws([&] { (void)checked_export(8, [](uint8_t *, size_t) -> size_t { throw std::runtime_error("export error"); }, limits); },
           "export exceptions propagate");
    require(checked_export(32, exporter, limits).size() == 32, "exact raw export size limit allowed");
    for (int mode = 0; mode < 3; ++mode) {
        bool active = true;
        size_t clears = 0;
        std::vector<std::string> order;
        throws([&] {
            checked_restore({1, 2, 3}, [&] { active = false; ++clears; order.push_back("clear"); },
                            [&](const uint8_t *, size_t size) -> size_t {
                require(!active, "failed importer still observes cleared initial state");
                active = true; // Simulate partially restored state that must not escape.
                order.push_back("import");
                if (mode == 0) return size - 1;
                if (mode == 1) return size + 1;
                throw std::runtime_error("injected import failure");
            });
        }, "short/long/throwing importer fails");
        require(!active && clears == 2, "restore failure clears partial state again");
        require(order == std::vector<std::string>({"clear", "import", "clear"}), "failure clear/import/clear order");
    }
    size_t clears = 0;
    size_t imports = 0;
    throws([&] { checked_restore({}, [&] { ++clears; }, [&](const uint8_t *, size_t n) { ++imports; return n; }); },
           "empty restore rejected");
    require(clears == 2 && imports == 0, "empty restore clears and never imports");
    clears = 0;
    throws([&] {
        checked_restore({1}, [&] { if (++clears == 1) throw std::runtime_error("clear failure"); },
                        [&](const uint8_t *, size_t n) { ++imports; return n; });
    }, "initial clear failure propagates");
    require(clears == 2 && imports == 0, "failed initial clear retried without importing");
    clears = 0;
    bool original_error = false;
    try {
        checked_restore({1}, [&] { if (++clears == 2) throw std::runtime_error("cleanup error"); },
                        [](const uint8_t *, size_t) -> size_t { throw std::runtime_error("primary import error"); });
    } catch (const std::runtime_error & error) { original_error = std::string(error.what()) == "primary import error"; }
    require(original_error && clears == 2, "cleanup failure does not hide primary restore failure");
}

void test_logits_and_crc() {
    detail::Crc64 crc;
    const std::string text = "123456789";
    crc.update(reinterpret_cast<const uint8_t *>(text.data()), 4);
    crc.update(reinterpret_cast<const uint8_t *>(text.data() + 4), text.size() - 4);
    require(crc.value() == 0x6c40df5f0b497347ULL, "table-based CRC64 matches ECMA known vector across chunks");
    require(reference_crc(reinterpret_cast<const uint8_t *>(text.data()), text.size()) == crc.value(), "independent CRC matches");
    auto result = compare_logits({0, 1, -2}, {0, 1, -2});
    require(result.passed && result.max_abs_error == 0 && result.max_rel_error == 0, "identical finite logits pass");
    require(compare_logits({0, 1, -2}, {0.00001f, 1.00001f, -2.00001f}).passed, "within default tolerance passes");
    require(compare_logits({100}, {101}, 0, 0.02).passed, "relative tolerance uses expected magnitude");
    result = compare_logits({1, 2}, {1.5f, 4}, 0, 0);
    require(!result.passed && result.max_abs_error == 2 && result.max_rel_error == 1, "error maxima cover all logits after failure");
    require(!compare_logits({0}, {0.001f}).passed, "absolute tolerance failure");
    result = compare_logits({0}, {0.00001f});
    require(result.passed && std::isinf(result.max_rel_error), "zero reference allows absolute tolerance but reports infinite relative error");
    require(!compare_logits({}, {}).passed, "empty logits rejected");
    require(!compare_logits({1}, {1, 2}).passed, "logit length mismatch rejected");
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    require(!compare_logits({nan}, {nan}).passed, "NaN/NaN cannot pass");
    require(!compare_logits({1}, {nan}).passed, "actual NaN rejected");
    require(!compare_logits({nan}, {1}).passed, "reference NaN rejected");
    require(!compare_logits({inf}, {inf}).passed, "equal positive infinities rejected");
    require(!compare_logits({-inf}, {-inf}).passed, "equal negative infinities rejected");
    require(!compare_logits({1}, {inf}).passed, "actual infinity rejected");
    require(!compare_logits({inf}, {1}).passed, "reference infinity rejected");
    require(!compare_logits({1}, {1}, -1, 0).passed, "negative tolerance rejected");
    require(!compare_logits({1}, {1}, 0, -1).passed, "negative relative tolerance rejected");
    require(!compare_logits({1}, {1}, nan, 0).passed, "NaN tolerance rejected");
    require(!compare_logits({1}, {1}, 0, inf).passed, "infinite tolerance rejected");
    require(same_tokens({1, 2, 3}, {1, 2, 3}), "same tokens pass");
    require(!same_tokens({1, 2, 3}, {1, 3, 2}), "token order mismatch rejected");
    require(!same_tokens({1, 2}, {1}), "token length mismatch rejected");
}

void require_close(double actual, double expected, const std::string & message) {
    require(std::isfinite(actual) && std::isfinite(expected) &&
            std::abs(actual - expected) <= 1e-10 * std::max(1.0, std::abs(expected)), message);
}

bench::Observation measurement(bench::Mode mode, uint32_t trial, std::optional<double> value, bool passed = true) {
    const uint32_t order = (3 + static_cast<uint32_t>(mode) - trial % 3) % 3;
    return {{bench::Phase::measurement, trial, order, mode}, passed, value};
}

void require_summary(const bench::Summary & summary, const std::string & status, uint32_t expected, uint32_t observed) {
    require(summary.status == status, "summary status is " + status);
    require(summary.n_expected == expected && summary.n_observed == observed, "summary preserves expected/unique attempted counts");
    require(summary.mean.has_value() == (status == "pass") && summary.stddev.has_value() == (status == "pass"),
            "only complete passed summaries expose mean/stddev");
}

void test_benchmark_schedule() {
    using bench::Mode;
    using bench::Phase;
    require(std::string(bench::mode_name(Mode::recompute)) == "recompute", "recompute label");
    require(std::string(bench::mode_name(Mode::dram)) == "dram", "DRAM label");
    require(std::string(bench::mode_name(Mode::buffered_file)) == "buffered-file", "buffered-file label remains honest");
    require(std::string(bench::phase_name(Phase::validation)) == "validation" &&
            std::string(bench::phase_name(Phase::warmup)) == "warmup" &&
            std::string(bench::phase_name(Phase::measurement)) == "measurement", "phase labels");
    throws([] { (void)bench::mode_name(static_cast<Mode>(99)); }, "invalid mode rejected");
    throws([] { (void)bench::phase_name(static_cast<Phase>(99)); }, "invalid phase rejected");
    bench::validate_counts(1, 1);
    bench::validate_counts(1000, 1000);
    for (const auto & counts : std::vector<std::pair<uint32_t, uint32_t>>{
             {0, 1}, {1, 0}, {1001, 1}, {1, 1001}, {UINT32_MAX, 1}, {1, UINT32_MAX}}) {
        bool expected_message = false;
        try { bench::validate_counts(counts.first, counts.second); }
        catch (const std::invalid_argument & error) {
            expected_message = std::string(error.what()) == "repetitions and warmup must be in [1, 1000]";
        }
        require(expected_message, "count validation retains exact documented error");
        throws([&] { (void)bench::make_schedule(counts.first, counts.second); }, "invalid counts rejected before scheduling");
    }
    const auto schedule = bench::make_schedule(6, 3);
    require(schedule.size() == 27, "exact warmup and measurement schedule length; golden excluded");
    std::array<uint32_t, 3> warmup_counts{};
    std::array<uint32_t, 3> measurement_counts{};
    std::array<std::array<uint32_t, 3>, 3> positions{};
    for (size_t i = 0; i < schedule.size(); ++i) {
        const auto & trial = schedule[i];
        const bool warmup = i < 9;
        const uint32_t round = static_cast<uint32_t>((warmup ? i : i - 9) / 3);
        const uint32_t order = static_cast<uint32_t>(i % 3);
        require(trial.phase == (warmup ? Phase::warmup : Phase::measurement), "all warmups finish before any measurement");
        require(trial.trial == round && trial.order == order, "zero-based round/order reset at phase boundary");
        require(static_cast<uint32_t>(trial.mode) == (round + order) % 3, "round rotates starting mode deterministically");
        const size_t mode = static_cast<size_t>(trial.mode);
        if (warmup) ++warmup_counts[mode];
        else { ++measurement_counts[mode]; ++positions[mode][order]; }
    }
    for (size_t mode = 0; mode < 3; ++mode) {
        require(warmup_counts[mode] == 3 && measurement_counts[mode] == 6, "exact equal per-mode warmup/measurement counts");
        for (size_t order = 0; order < 3; ++order) require(positions[mode][order] == 2, "balanced measurement order positions");
    }
    const auto minimum = bench::make_schedule(1, 1);
    require(minimum.size() == 6 && minimum[3].phase == Phase::measurement && minimum[3].trial == 0,
            "minimum schedule includes one warmup and one measurement per mode");
    const auto maximum = bench::make_schedule(1000, 1000);
    require(maximum.size() == 6000 && maximum[2999].phase == Phase::warmup && maximum[3000].phase == Phase::measurement,
            "maximum accepted schedule remains bounded and phase separated");
}

void test_benchmark_rates() {
    using bench::Mode;
    const bench::Timings recompute{6, 4, 15, 18, 36, 45, 20};
    const auto full = bench::derive_rates(Mode::recompute, 8, 2, 4, recompute);
    require_close(full.actual_prefill_eval_tps, 1000, "recompute evaluates P+S tokens with ms-to-second conversion");
    require_close(full.effective_prompt_tps, 10000.0 / 15, "effective prompt includes full prompt and end-to-end ready delay");
    require_close(full.conditioned_decode_forward_tps.value(), 200, "conditioned decode uses D pure forward calls");
    require(full.generation_after_first_tps.has_value(), "D>1 has post-first generation throughput");
    require_close(*full.generation_after_first_tps, 3000.0 / 18, "post-first numerator D-1 and sample-to-sample denominator");
    require_close(full.end_to_end_output_tps, 4000.0 / 36, "end-to-end output denominator is last sample, not request end");
    const bench::Timings cached{0, 4, 12, 15, 30, 40, 20};
    for (Mode mode : {Mode::dram, Mode::buffered_file}) {
        const auto hit = bench::derive_rates(mode, 8, 2, 4, cached);
        require_close(hit.actual_prefill_eval_tps, 500, "hit evaluates only S tokens, not cached P");
        require_close(hit.effective_prompt_tps, 10000.0 / 12, "hit effective input still counts the full prompt");
        require_close(hit.conditioned_decode_forward_tps.value(), 200, "hit decode-forward denominator excludes sampling/validation");
        require_close(*hit.generation_after_first_tps, 200, "hit post-first rate spans sampling timestamps");
        require_close(hit.end_to_end_output_tps, 4000.0 / 30, "hit output counts D over end-to-end token-ready time");
    }
    bench::Timings single{6, 4, 12, 15, 15, 20, 4};
    const auto one = bench::derive_rates(Mode::recompute, 8, 2, 1, single);
    require(!one.generation_after_first_tps, "D=1 post-first generation rate is NA, never zero or infinity");
    require_close(one.conditioned_decode_forward_tps.value(), 250, "D=1 retains its final sampled-token forward update");
    require_close(one.end_to_end_output_tps, 1000.0 / 15, "D=1 uses first/last sample, not the later final forward end");
    single.request_ms = 100;
    const auto later_end = bench::derive_rates(Mode::recompute, 8, 2, 1, single);
    require_close(later_end.end_to_end_output_tps, one.end_to_end_output_tps, "extra end-of-request host work does not move last sample");
    require_close(later_end.conditioned_decode_forward_tps.value(), one.conditioned_decode_forward_tps.value(),
                  "pure forward throughput is not renamed end-to-end decode time");
    const auto wide = bench::derive_rates(Mode::recompute, UINT32_MAX, UINT32_MAX, UINT32_MAX, recompute);
    require_close(wide.actual_prefill_eval_tps, 1000.0 * (2ULL * UINT32_MAX) / 10, "P+S widened before addition");
    require_close(wide.conditioned_decode_forward_tps.value(), 1000.0 * UINT32_MAX / 20, "large positive D never wraps");
    for (const auto & counts : std::vector<std::array<uint32_t, 3>>{{{0, 2, 4}}, {{8, 0, 4}}}) {
        throws([&] { (void)bench::derive_rates(Mode::recompute, counts[0], counts[1], counts[2], recompute); },
               "zero prefix or suffix token counts rejected");
    }
    const auto eight = bench::derive_rates(Mode::recompute, 8, 2, 8, recompute);
    require_close(eight.conditioned_decode_forward_tps.value(), 400, "D=8 retains all eight pure forward updates");
    require_close(eight.generation_after_first_tps.value(), 7000.0 / 18, "D=8 post-first numerator remains seven");
    require_close(eight.end_to_end_output_tps, 8000.0 / 36, "D=8 sampled-output numerator remains eight");
    const std::array<double bench::Timings::*, 7> fields{{
        &bench::Timings::prefix_ms, &bench::Timings::tail_ms, &bench::Timings::prompt_ready_ms,
        &bench::Timings::ttft_ms, &bench::Timings::last_sample_ms, &bench::Timings::request_ms,
        &bench::Timings::decode_forward_ms}};
    for (auto field : fields) {
        for (double invalid : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
            auto bad = recompute;
            bad.*field = invalid;
            throws([&] { (void)bench::derive_rates(Mode::recompute, 8, 2, 4, bad); }, "invalid required elapsed time cannot yield rates");
        }
    }
    const auto invalid_boundaries = [&](const bench::Timings & t, uint32_t D = 4) {
        throws([&] { (void)bench::derive_rates(Mode::recompute, 8, 2, D, t); }, "inconsistent timing boundaries rejected");
    };
    auto bad = recompute; bad.prompt_ready_ms = 9; invalid_boundaries(bad);
    bad = recompute; bad.ttft_ms = 14; invalid_boundaries(bad);
    bad = recompute; bad.last_sample_ms = 17; invalid_boundaries(bad);
    bad = recompute; bad.last_sample_ms = bad.ttft_ms; invalid_boundaries(bad);
    bad = recompute; bad.request_ms = bad.last_sample_ms; invalid_boundaries(bad);
    bad = recompute; bad.request_ms = bad.last_sample_ms - 1; invalid_boundaries(bad);
    bad = recompute; bad.decode_forward_ms = bad.request_ms - bad.ttft_ms + 1; invalid_boundaries(bad);
    invalid_boundaries(recompute, 1);
    bad = cached; bad.prefix_ms = 1;
    throws([&] { (void)bench::derive_rates(Mode::dram, 8, 2, 4, bad); }, "cache path cannot claim prefix evaluation time");
    bad = cached; bad.prefix_ms = -1;
    throws([&] { (void)bench::derive_rates(Mode::buffered_file, 8, 2, 4, bad); }, "negative cache prefix time rejected");
    bad = cached; bad.tail_ms = std::numeric_limits<double>::denorm_min();
    throws([&] { (void)bench::derive_rates(Mode::dram, 8, 2, 4, bad); }, "finite tiny denominator yielding infinite rate rejected");
    throws([&] { (void)bench::derive_rates(static_cast<Mode>(99), 8, 2, 4, recompute); }, "unknown mode cannot yield rates");
}

void test_prefill_only_rates() {
    using bench::Mode;
    static_assert(std::is_same<decltype(bench::Rates{}.conditioned_decode_forward_tps), std::optional<double>>::value,
                  "decode-forward throughput must represent pure-PP NA without NaN");
    const bench::Timings pure{6, 4, 15, 18, 18, 18, 0};
    for (Mode mode : {Mode::recompute, Mode::dram, Mode::buffered_file}) {
        auto timings = pure;
        if (mode != Mode::recompute) timings.prefix_ms = 0;
        const auto rates = bench::derive_rates(mode, 262143, 1, 0, timings);
        require_close(rates.actual_prefill_eval_tps, mode == Mode::recompute ? 26214400.0 : 250.0,
                      "pure pp262144 retains full input and only evaluates suffix on a hit");
        require_close(rates.effective_prompt_tps, 262144000.0 / 15, "pure pp262144 effective prompt preserves exact P+S");
        require(!rates.conditioned_decode_forward_tps && !rates.generation_after_first_tps,
                "D=0 decode metrics are optional NA, never zero or NaN rates");
        require_close(rates.end_to_end_output_tps, 1000.0 / 18, "D=0 output rate counts one sampled first token, not zero");
        timings.request_ms = 100;
        const auto later = bench::derive_rates(mode, 262143, 1, 0, timings);
        require_close(later.end_to_end_output_tps, rates.end_to_end_output_tps, "D=0 post-sample host work does not change TTFT output denominator");
        std::vector<bench::Observation> forward, generation;
        for (uint32_t trial = 0; trial < 2; ++trial) {
            forward.push_back(measurement(mode, trial, rates.conditioned_decode_forward_tps));
            generation.push_back(measurement(mode, trial, rates.generation_after_first_tps));
        }
        require_summary(bench::summarize(mode, 2, forward, true), "not_applicable", 2, 2);
        require_summary(bench::summarize(mode, 2, generation, true), "not_applicable", 2, 2);
        require_summary(bench::summarize(mode, 2, forward), "failed", 2, 2);
        require_summary(bench::summarize(mode, 2, forward, true, true), "failed", 2, 2);
        forward[0].value = std::numeric_limits<double>::quiet_NaN();
        require_summary(bench::summarize(mode, 2, forward, true), "failed", 2, 2);
    }
    const std::array<double bench::Timings::*, 6> fields{{
        &bench::Timings::prefix_ms, &bench::Timings::tail_ms, &bench::Timings::prompt_ready_ms,
        &bench::Timings::ttft_ms, &bench::Timings::last_sample_ms, &bench::Timings::request_ms}};
    for (auto field : fields) {
        for (double invalid : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
            auto bad = pure; bad.*field = invalid;
            throws([&] { (void)bench::derive_rates(Mode::recompute, 8, 2, 0, bad); }, "D=0 retains positive finite prefill/sample timing checks");
        }
    }
    for (double invalid : {-1.0, 1.0, std::numeric_limits<double>::denorm_min(),
                           std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        auto bad = pure; bad.decode_forward_ms = invalid;
        throws([&] { (void)bench::derive_rates(Mode::recompute, 8, 2, 0, bad); }, "D=0 forward time must be exactly zero, not a nonfinite NA marker");
    }
    const auto invalid_boundary = [&](const bench::Timings & timings) {
        throws([&] { (void)bench::derive_rates(Mode::recompute, 8, 2, 0, timings); }, "D=0 rejects inconsistent timing boundaries");
    };
    auto bad = pure; bad.prompt_ready_ms = 9; invalid_boundary(bad);
    bad = pure; bad.ttft_ms = 14; invalid_boundary(bad);
    bad = pure; bad.last_sample_ms = 17; invalid_boundary(bad);
    bad = pure; bad.last_sample_ms = 19; bad.request_ms = 20; invalid_boundary(bad);
    bad = pure; bad.request_ms = 17; invalid_boundary(bad);
    throws([&] { (void)bench::derive_rates(Mode::dram, 8, 2, 0, pure); }, "D=0 cache still requires zero prefix evaluation time");
    throws([&] { (void)bench::derive_rates(Mode::recompute, 0, 2, 0, pure); }, "D=0 does not permit an empty prefix");
    throws([&] { (void)bench::derive_rates(Mode::recompute, 8, 0, 0, pure); }, "D=0 still requires an unevaluated suffix");
    for (uint32_t D : {1U, 8U}) {
        throws([&] { (void)bench::derive_rates(Mode::recompute, 8, 2, D, pure); }, "D>0 cannot use zero forward time");
        bench::Timings timings{6, 4, 15, 18, D == 1 ? 18.0 : 36.0, 45, 20};
        const auto positive = bench::derive_rates(Mode::recompute, 8, 2, D, timings);
        require(positive.conditioned_decode_forward_tps.has_value(), "D=1/D=8 must expose positive forward throughput");
        require_close(positive.conditioned_decode_forward_tps.value(), 1000.0 * D / 20, "D>0 retains its exact D-forward numerator");
        require_summary(bench::summarize(Mode::recompute, 1,
            {measurement(Mode::recompute, 0, positive.conditioned_decode_forward_tps)}), "pass", 1, 1);
        require_summary(bench::summarize(Mode::recompute, 1,
            {measurement(Mode::recompute, 0, std::nullopt)}), "failed", 1, 1);
        timings.request_ms = timings.last_sample_ms;
        throws([&] { (void)bench::derive_rates(Mode::recompute, 8, 2, D, timings); }, "D>0 still needs request time after final sample");
    }
}

void test_benchmark_summaries() {
    using bench::Mode;
    using bench::Phase;
    const std::vector<bench::Observation> complete{
        measurement(Mode::recompute, 0, 10), measurement(Mode::recompute, 1, 20), measurement(Mode::recompute, 2, 30)};
    auto summary = bench::summarize(Mode::recompute, 3, complete);
    require_summary(summary, "pass", 3, 3);
    require_close(*summary.mean, 20, "per-trial arithmetic mean");
    require_close(*summary.stddev, 10, "sample standard deviation uses n-1");
    summary = bench::summarize(Mode::dram, 1, {measurement(Mode::dram, 0, 7)});
    require_summary(summary, "pass", 1, 1);
    require_close(*summary.mean, 7, "singleton mean");
    require_close(*summary.stddev, 0, "singleton standard deviation is zero, matching llama-bench");
    auto excluded = complete;
    excluded.push_back({{Phase::validation, 0, 0, Mode::recompute}, false, std::numeric_limits<double>::infinity()});
    excluded.push_back({{Phase::warmup, 0, 0, Mode::recompute}, false, std::nullopt});
    excluded.push_back(measurement(Mode::dram, 0, std::numeric_limits<double>::quiet_NaN(), false));
    summary = bench::summarize(Mode::recompute, 3, excluded);
    require_summary(summary, "pass", 3, 3);
    require_close(*summary.mean, 20, "golden/warmup/other modes never enter this mean");
    std::reverse(excluded.begin(), excluded.end());
    summary = bench::summarize(Mode::recompute, 3, excluded);
    require_summary(summary, "pass", 3, 3);
    require_close(*summary.stddev, 10, "observation storage order does not replace recorded trial order");
    const bench::Timings quick{80, 20, 110, 111, 141, 171, 40};
    const bench::Timings slow{160, 40, 210, 211, 241, 271, 40};
    const double quick_tps = bench::derive_rates(Mode::recompute, 8, 2, 2, quick).actual_prefill_eval_tps;
    const double slow_tps = bench::derive_rates(Mode::recompute, 8, 2, 2, slow).actual_prefill_eval_tps;
    summary = bench::summarize(Mode::recompute, 2, {measurement(Mode::recompute, 0, quick_tps), measurement(Mode::recompute, 1, slow_tps)});
    require_summary(summary, "pass", 2, 2);
    require_close(*summary.mean, 75, "mean of per-trial rates, not tokens divided by mean time");
    require(std::abs(*summary.mean - 10000.0 / 150) > 1, "ratio of means is deliberately a different statistic");
    const auto failed = [&](const std::vector<bench::Observation> & rows, uint32_t observed = 3) {
        require_summary(bench::summarize(Mode::recompute, 3, rows), "failed", 3, observed);
    };
    auto malformed = complete; malformed.push_back(complete[0]); failed(malformed);
    malformed = complete; malformed[1].trial.order = (malformed[1].trial.order + 1) % 3; failed(malformed);
    malformed = complete; malformed[1].trial.order = UINT32_MAX; failed(malformed);
    malformed = complete; malformed[1].trial.trial = UINT32_MAX; failed(malformed);
    malformed = complete; malformed.push_back(measurement(Mode::recompute, 3, 40)); failed(malformed, 4);
    malformed = complete; malformed[1].passed = false; failed(malformed);
    malformed = complete; malformed[1].value = std::numeric_limits<double>::quiet_NaN(); failed(malformed);
    malformed = complete; malformed[1].value = std::numeric_limits<double>::infinity(); failed(malformed);
    malformed = complete; malformed[1].value.reset(); failed(malformed);
    auto missing = complete; missing.pop_back();
    require_summary(bench::summarize(Mode::recompute, 3, missing), "incomplete", 3, 2);
    require_summary(bench::summarize(Mode::recompute, 3, {}), "incomplete", 3, 0);
    missing[0].passed = false; failed(missing, 2);
    std::vector<bench::Observation> all_na{
        measurement(Mode::recompute, 0, std::nullopt), measurement(Mode::recompute, 1, std::nullopt),
        measurement(Mode::recompute, 2, std::nullopt)};
    require_summary(bench::summarize(Mode::recompute, 3, all_na, true), "not_applicable", 3, 3);
    failed(all_na);
    auto mixed = all_na; mixed[0].value = 10;
    require_summary(bench::summarize(Mode::recompute, 3, mixed, true), "failed", 3, 3);
    mixed = all_na; mixed[1].passed = false;
    require_summary(bench::summarize(Mode::recompute, 3, mixed, true), "failed", 3, 3);
    mixed = all_na; mixed.push_back(all_na[0]);
    require_summary(bench::summarize(Mode::recompute, 3, mixed, true), "failed", 3, 3);
    mixed = all_na; mixed.pop_back();
    require_summary(bench::summarize(Mode::recompute, 3, mixed, true), "incomplete", 3, 2);
    require_summary(bench::summarize(Mode::recompute, 3, complete, false, true), "failed", 3, 3);
    require_summary(bench::summarize(Mode::recompute, 3, {}, false, true), "failed", 3, 0);
    require_summary(bench::summarize(Mode::recompute, 3, all_na, true, true), "failed", 3, 3);
    const double largest = std::numeric_limits<double>::max();
    summary = bench::summarize(Mode::recompute, 2, {measurement(Mode::recompute, 0, 0), measurement(Mode::recompute, 1, largest)});
    require_summary(summary, "pass", 2, 2);
    require_close(*summary.mean, largest / 2, "finite large mean avoids double-sum overflow");
    require_close(*summary.stddev, largest / std::sqrt(2.0), "finite large sample stddev avoids squared-double overflow");
    require_summary(bench::summarize(Mode::recompute, 2, {
        measurement(Mode::recompute, 0, -largest), measurement(Mode::recompute, 1, largest)}), "failed", 2, 2);
    throws([&] { (void)bench::summarize(Mode::recompute, 0, complete); }, "zero expected count rejected");
    throws([&] { (void)bench::summarize(Mode::recompute, 1001, complete); }, "excessive expected count rejected");
    throws([&] { (void)bench::summarize(static_cast<Mode>(99), 3, complete); }, "invalid requested summary mode rejected");
}

void test_repeated_fake_benchmark(const fs::path & run) {
    using bench::Mode;
    using bench::Phase;
    const std::vector<int32_t> prefix_a{3, 7, 11, 5};
    const std::vector<int32_t> prefix_b{9, 1, 8, 2};
    const std::vector<int32_t> suffix{4, 6};
    constexpr uint32_t D = 3;
    FakeModel golden;
    golden.prefill(prefix_a);
    golden.prefill(suffix);
    std::vector<std::vector<float>> golden_before, golden_after;
    std::vector<int32_t> golden_tokens;
    for (uint32_t i = 0; i < D; ++i) {
        golden_before.push_back(golden.logits());
        const auto & logits = golden_before.back();
        const int32_t token = static_cast<int32_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
        golden_tokens.push_back(token);
        golden.step(token, golden.history.size());
        golden_after.push_back(golden.logits());
    }
    FakeModel model;
    model.prefill(prefix_a);
    const auto original = model.bytes();
    const auto state = checked_export(original.size(), [&](uint8_t * destination, size_t size) {
        std::copy(original.begin(), original.end(), destination);
        return size;
    }, small_limits());
    const Identity identity{"repeated-fake-complete-state;model=none;backend=host", prefix_a, prefix_a.size()};
    MemoryStore memory(small_limits());
    const fs::path root = run / "repeated-fake-benchmark";
    BufferedFileStore files(root, small_limits());
    // Exactly one publication per backend before any warmup/measurement.
    memory.put("repeat-A", {identity, state});
    files.put("repeat-A", {identity, state});
    const auto retained = memory.get("repeat-A", identity).snapshot;
    const auto published = read_fixture(root / "repeat-A.qcache");
    std::vector<bench::Observation> observations{{{Phase::validation, 0, 0, Mode::recompute}, true, 999999}};
    std::array<uint32_t, 3> hits{};
    for (const auto & trial : bench::make_schedule(3, 2)) {
        std::vector<std::string> order;
        model.clear();
        model.prefill(prefix_b);
        order.push_back("B");
        require(model.bytes() != state, "every warmup/measurement has actual B interference before request reset");
        if (trial.mode == Mode::recompute) {
            model.clear(); order.push_back("clear");
            model.prefill(prefix_a); order.push_back("prefix");
        } else {
            const auto hit = trial.mode == Mode::dram ? memory.get("repeat-A", identity) : files.get("repeat-A", identity);
            require_status(hit, LookupStatus::hit, "repeated cache hit uses unchanged identity and complete bytes");
            require(hit.snapshot->state == state, "repeated cache hit returns original published bytes");
            if (trial.mode == Mode::dram) require(hit.snapshot.get() == retained.get(), "all DRAM trials reuse one immutable published object");
            checked_restore(hit.snapshot->state, [&] { model.clear(); order.push_back("clear"); },
                            [&](const uint8_t * data, size_t size) { order.push_back("restore"); return model.restore(data, size); });
        }
        require(model.history == prefix_a && model.bytes() == state, "every trial begins suffix at exact restored/recomputed prefix boundary");
        model.prefill(suffix); order.push_back("suffix");
        require(order == std::vector<std::string>({"B", "clear", trial.mode == Mode::recompute ? "prefix" : "restore", "suffix"}),
                "every trial preserves B/reset/restore-or-prefix/suffix protocol");
        std::vector<int32_t> tokens;
        for (uint32_t i = 0; i < D; ++i) {
            const auto logits = model.logits();
            require(compare_logits(golden_before[i], logits, 0, 0).passed, "every warmup/measurement checks finite full logits before sampling");
            const int32_t token = static_cast<int32_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
            tokens.push_back(token);
            model.step(token, model.history.size()); // Includes the forward after the final sample.
            require(compare_logits(golden_after[i], model.logits(), 0, 0).passed, "every forward including the last is checked against golden");
        }
        require(same_tokens(tokens, golden_tokens), "every warmup/measurement checks greedy token parity");
        require(model.history.size() == prefix_a.size() + suffix.size() + D, "D samples retain D forward updates including final token");
        ++hits[static_cast<size_t>(trial.mode)];
        const bench::Timings timings{trial.mode == Mode::recompute ? 6.0 : 0.0, 4, 15, 18, 36, 45, 20};
        const auto rates = bench::derive_rates(trial.mode, prefix_a.size(), suffix.size(), D, timings);
        observations.push_back({trial, true, rates.actual_prefill_eval_tps});
    }
    for (Mode mode : {Mode::recompute, Mode::dram, Mode::buffered_file}) {
        require(hits[static_cast<size_t>(mode)] == 5, "each mode completes two warmups and three measurements");
        require_summary(bench::summarize(mode, 3, observations), "pass", 3, 3);
    }
    require(read_fixture(root / "repeat-A.qcache") == published, "file cache remains exactly the once-published snapshot across all trials");
    require(std::distance(fs::directory_iterator(root), fs::directory_iterator{}) == 1, "repeated hits never republish or leave staging files");
    require(memory.get("repeat-A", identity).snapshot.get() == retained.get(), "DRAM entry was never replaced between trials");
}

void test_prefill_only_fake(const fs::path & run) {
    using bench::Mode;
    const std::vector<int32_t> prefix_a{3, 7, 11, 5};
    const std::vector<int32_t> prefix_b{9, 1, 8, 2};
    const std::vector<int32_t> suffix{4};
    const auto sample_first = [](const FakeModel & model) {
        const auto logits = model.logits();
        return std::vector<int32_t>{static_cast<int32_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())))};
    };
    FakeModel golden;
    golden.prefill(prefix_a);
    golden.prefill(suffix);
    const auto golden_logits = golden.logits();
    const auto golden_tokens = sample_first(golden);
    require(golden_tokens.size() == 1 && golden.history.size() == prefix_a.size() + suffix.size(),
            "pure-PP golden samples one first token without forwarding it");
    FakeModel model;
    model.prefill(prefix_a);
    const auto prefix_state = model.bytes();
    const Identity identity{"pure-prefill-fake;model=none;backend=host", prefix_a, prefix_a.size()};
    MemoryStore memory(small_limits());
    const fs::path root = run / "pure-prefill-fake";
    BufferedFileStore files(root, small_limits());
    memory.put("pure-A", {identity, prefix_state});
    files.put("pure-A", {identity, prefix_state});
    const auto published = read_fixture(root / "pure-A.qcache");
    for (const auto & trial : bench::make_schedule(2, 1)) {
        model.clear();
        model.prefill(prefix_b);
        require(model.bytes() != prefix_state, "D=0 warmup/measurement still replaces active A with B");
        std::vector<std::string> order{"B"};
        if (trial.mode == Mode::recompute) {
            model.clear(); order.push_back("clear");
            model.prefill(prefix_a); order.push_back("prefix");
        } else {
            const auto hit = trial.mode == Mode::dram ? memory.get("pure-A", identity) : files.get("pure-A", identity);
            require_status(hit, LookupStatus::hit, "D=0 retrieves once-published prefix snapshot");
            checked_restore(hit.snapshot->state, [&] { model.clear(); order.push_back("clear"); },
                [&](const uint8_t * state, size_t size) { order.push_back("restore"); return model.restore(state, size); });
        }
        require(model.history == prefix_a && model.bytes() == prefix_state, "D=0 restore ends at exact prefix boundary before suffix replay");
        model.prefill(suffix); order.push_back("suffix");
        require(compare_logits(golden_logits, model.logits(), 0, 0).passed, "D=0 replays S and compares complete finite suffix logits");
        const auto before_sample = model.bytes();
        const size_t position = model.history.size();
        const auto tokens = sample_first(model); order.push_back("sample");
        require(tokens.size() == 1 && same_tokens(tokens, golden_tokens), "D=0 samples exactly one parity-matching first token");
        require(model.history.size() == position && position == prefix_a.size() + suffix.size() && model.bytes() == before_sample,
                "D=0 performs zero forwards after sampling and leaves all inference state/positions unchanged");
        require(order == std::vector<std::string>({"B", "clear", trial.mode == Mode::recompute ? "prefix" : "restore", "suffix", "sample"}),
                "D=0 request order retains clear/restore/suffix/sample without a decode update");
    }
    require(read_fixture(root / "pure-A.qcache") == published &&
            std::distance(fs::directory_iterator(root), fs::directory_iterator{}) == 1,
            "D=0 repeated warmups/measurements do not republish cache entries");
    // Probe argument validation owns actual context-cap rejection. This fake
    // boundary case proves that sampling itself consumes no additional slot.
    constexpr size_t context = 262144;
    FakeModel full;
    full.prefill(std::vector<int32_t>(context - 1, 3));
    full.prefill({4});
    const auto before_sample = full.bytes();
    const auto tokens = sample_first(full);
    require(tokens.size() == 1 && full.history.size() == context && full.bytes() == before_sample,
            "P262143+S1 fills context262144 and D0 sampling uses no extra context position");
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--child-read") return child_read(argv[2]);
        if (argc > 2) throw std::invalid_argument("usage: test_snapshot_store [temp_root]");
        const fs::path parent = argc == 2 ? fs::path(argv[1]) : artifact_root / "tests";
        RunDirectory run(parent);
        std::cout << "Host-only snapshot tests; no llama/CUDA linkage; no model loading or GPU work.\n"
                  << "Owned fixtures: " << run.path << '\n';
        const auto group = [](const char * label, const std::function<void()> & test) {
            test();
            std::cout << "[PASS] " << label << '\n';
        };
        group("DRAM immutable roundtrips / identity / duplicates / input bounds", test_memory);
        group("buffered-file roundtrips / canonical LE / streamed data", [&] { test_files(run.path); });
        group("corruption / truncation / trailing bytes / versions / length bounds", [&] { test_corruption(run.path); });
        group("quota / disk reserve / staging / atomic publication / I/O failure cleanup", [&] {
            test_quota_and_staging(run.path);
            test_atomic_failure_cleanup(run.path);
        });
        group("owned paths / permissions / symlink-hardlink-FIFO rejection", [&] { test_path_safety(run.path); });
        group("persistent state survives fresh process", [&] { test_fresh_process(run.path); });
        group("fake A/B/A / suffix progression / checked export-restore / failure clears", [&] { test_state_protocol(run.path); });
        group("finite logits / tolerances / token equality / CRC64 known vector", test_logits_and_crc);
        std::cout << "preserved storage/protocol baseline: " << checks << " parent-process checks\n";
        group("benchmark schedule / phase separation / balanced rotation / count bounds", test_benchmark_schedule);
        group("benchmark rates / units / numerators / D=1 / token-ready boundaries", test_benchmark_rates);
        group("per-trial mean / sample stddev / complete sets / failure and NA summaries", test_benchmark_summaries);
        group("repeated fake A/B/A / once-published stores / warmup and measurement parity", [&] { test_repeated_fake_benchmark(run.path); });
        group("pure-PP D=0 / optional decode rates / first-token parity / full-input sampling", [&] {
            test_prefill_only_rates();
            test_prefill_only_fake(run.path);
        });
        std::cout << "host tests passed: 13 groups, " << checks << " parent-process checks\n"
                  << "new benchmark model parity/performance/256K: not run; no-model host tests only; prior run history unchanged\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "host tests FAILED: " << error.what() << '\n'
                  << "model parity/performance/256K: not run; no model loaded\n";
        return 1;
    }
}
