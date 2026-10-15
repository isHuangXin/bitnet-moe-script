#ifndef QWEN36_PREFIX_SNAPSHOT_STORE_HPP
#define QWEN36_PREFIX_SNAPSHOT_STORE_HPP

// Host-only, single-writer prefix snapshots. Not an importer for untrusted files.
// CRC64 detects accidental damage; it does not authenticate model state.
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <dirent.h>
#include <fcntl.h>
#include <linux/fs.h>
#include <liburing.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace qwen_cache {

struct Limits {
    uint64_t max_blob_bytes = 8ULL << 30;
    uint64_t max_store_bytes = 16ULL << 30;
    uint64_t disk_reserve_bytes = 16ULL << 30;
};

struct Identity {
    std::string configuration;
    std::vector<int32_t> prefix_tokens;
    uint64_t processed_tokens = 0;

    bool operator==(const Identity & other) const {
        return configuration == other.configuration &&
               prefix_tokens == other.prefix_tokens &&
               processed_tokens == other.processed_tokens;
    }
};

struct Snapshot {
    Identity identity;
    std::vector<uint8_t> state;
};

enum class LookupStatus { hit, miss, incompatible, corrupt };

struct LookupResult {
    LookupStatus status;
    std::shared_ptr<const Snapshot> snapshot;
    std::string detail;
};

namespace detail {

// Canonical v1 layout (all integers LE): magic[8], version:u32, header_size:u32,
// configuration_bytes:u64, token_count:u64, processed_tokens:u64,
// state_bytes:u64, record_bytes:u64, crc64:u64; then configuration, i32 tokens,
// opaque complete state. CRC covers header[0:56] and every payload byte.
inline constexpr size_t header_bytes = 64;
inline constexpr uint64_t max_configuration_bytes = 1ULL << 20;
inline constexpr uint64_t max_prefix_tokens = 1ULL << 20;
inline constexpr std::array<uint8_t, 8> magic = {{'Q', 'W', 'N', 'S', 'N', 'A', 'P', 0}};

inline std::runtime_error io_error(const std::string & operation, int error = errno) {
    return std::runtime_error(operation + ": " + std::strerror(error));
}

inline Limits valid_limits(Limits limits) {
    if (limits.max_blob_bytes == 0 || limits.max_store_bytes == 0) {
        throw std::invalid_argument("snapshot byte limits must be nonzero");
    }
    return limits;
}

inline void valid_key(const std::string & key) {
    if (key.empty() || key.size() > 128) {
        throw std::invalid_argument("snapshot key must contain 1..128 ASCII letters/digits/_/-");
    }
    for (unsigned char c : key) {
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '_' || c == '-')) {
            throw std::invalid_argument("unsafe snapshot key");
        }
    }
}

inline uint64_t record_size(uint64_t configuration, uint64_t tokens,
                            uint64_t processed, uint64_t state, const Limits & limits) {
    if (configuration == 0 || configuration > max_configuration_bytes ||
        tokens > max_prefix_tokens || processed != tokens || state == 0) {
        throw std::invalid_argument("invalid identity lengths, prefix boundary, or empty state");
    }
    // The preceding caps make the metadata addition and multiplication safe.
    const uint64_t overhead = header_bytes + configuration + tokens * 4;
    if (state > limits.max_blob_bytes || overhead > limits.max_blob_bytes - state ||
        state > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::length_error("snapshot exceeds max_blob_bytes or addressable memory");
    }
    return overhead + state;
}

inline uint64_t record_size(const Snapshot & snapshot, const Limits & limits) {
    return record_size(snapshot.identity.configuration.size(), snapshot.identity.prefix_tokens.size(),
                       snapshot.identity.processed_tokens, snapshot.state.size(), limits);
}

inline void check_quota(uint64_t used, uint64_t added, const Limits & limits) {
    if (used > limits.max_store_bytes || added > limits.max_store_bytes - used) {
        throw std::length_error("snapshot store quota exceeded (including staging bytes)");
    }
}

inline void put_le(uint8_t * out, uint64_t value, size_t bytes) {
    for (size_t i = 0; i < bytes; ++i) {
        out[i] = static_cast<uint8_t>(value & 0xff);
        value >>= 8;
    }
}

inline uint64_t get_le(const uint8_t * in, size_t bytes) {
    uint64_t value = 0;
    for (size_t i = 0; i < bytes; ++i) {
        value |= static_cast<uint64_t>(in[i]) << (i * 8);
    }
    return value;
}

class Crc64 {
public:
    void update(const uint8_t * bytes, size_t size) {
        // CRC-64/ECMA-182, non-reflected, initial value and xorout both zero.
        static const std::array<uint64_t, 256> table = [] {
            std::array<uint64_t, 256> result{};
            for (size_t i = 0; i < result.size(); ++i) {
                uint64_t crc = static_cast<uint64_t>(i) << 56;
                for (int bit = 0; bit < 8; ++bit) {
                    crc = (crc & (1ULL << 63)) ? (crc << 1) ^ 0x42f0e1eba9ea3693ULL : crc << 1;
                }
                result[i] = crc;
            }
            return result;
        }();
        for (size_t i = 0; i < size; ++i) {
            value_ = table[static_cast<uint8_t>((value_ >> 56) ^ bytes[i])] ^ (value_ << 8);
        }
    }
    uint64_t value() const { return value_; }

private:
    uint64_t value_ = 0;
};

class Fd {
public:
    explicit Fd(int fd = -1) : fd_(fd) {}
    ~Fd() { if (fd_ >= 0) ::close(fd_); }
    Fd(const Fd &) = delete;
    Fd & operator=(const Fd &) = delete;
    Fd(Fd && other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
    Fd & operator=(Fd && other) noexcept {
        if (this != &other) {
            if (fd_ >= 0) ::close(fd_);
            fd_ = std::exchange(other.fd_, -1);
        }
        return *this;
    }
    int get() const { return fd_; }
    int release() { return std::exchange(fd_, -1); }

private:
    int fd_;
};

inline Fd open_root(const std::filesystem::path & path) {
    if (path.empty() || path.native().find('\0') != std::string::npos) {
        throw std::invalid_argument("empty or invalid buffered-file root");
    }
    // Walk each component, rather than following a symlink in an ancestor.
    for (const auto & component : path) {
        if (component == "..") throw std::invalid_argument("parent traversal in buffered-file root");
    }
    Fd current(::open(path.is_absolute() ? "/" : ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC));
    if (current.get() < 0) throw io_error("open root anchor");
    for (const auto & component : path) {
        if (component.empty() || component == "/" || component == ".") continue;
        int next = ::openat(current.get(), component.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        if (next < 0 && errno == ENOENT) {
            if (::mkdirat(current.get(), component.c_str(), 0700) != 0 && errno != EEXIST) {
                throw io_error("create buffered-file directory");
            }
            next = ::openat(current.get(), component.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        }
        if (next < 0) throw io_error("open buffered-file directory without symlinks");
        current = Fd(next);
    }
    struct stat st{};
    if (::fstat(current.get(), &st) != 0) throw io_error("stat buffered-file root");
    if (!S_ISDIR(st.st_mode) || st.st_uid != ::geteuid() || (st.st_mode & 0777) != 0700) {
        throw std::runtime_error("buffered-file root must be locally owned and mode 0700");
    }
    return current;
}

inline void check_file(const struct stat & st) {
    if (!S_ISREG(st.st_mode) || st.st_uid != ::geteuid() ||
        (st.st_mode & 0777) != 0600 || st.st_nlink != 1 || st.st_size < 0) {
        throw std::runtime_error("snapshot must be a locally owned mode-0600 regular file with one link");
    }
}

inline void read_all(int fd, uint8_t * data, size_t size) {
    while (size != 0) {
        const ssize_t n = ::read(fd, data, std::min<size_t>(size, 1U << 20));
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) throw io_error("read buffered-file snapshot");
        if (n == 0) throw std::runtime_error("truncated buffered-file snapshot");
        data += n;
        size -= static_cast<size_t>(n);
    }
}

inline constexpr size_t direct_alignment = 4096;
inline constexpr size_t direct_chunk_bytes = 1U << 20;
inline constexpr unsigned direct_queue_depth = 8;

class DirectReader {
public:
    DirectReader(int fd, uint64_t file_size) : fd_(fd), file_size_(file_size) {
        if (file_size == 0) throw std::runtime_error("empty direct-I/O snapshot");
        capacity_ = static_cast<size_t>(std::min<uint64_t>(file_size, direct_chunk_bytes * direct_queue_depth));
        capacity_ = (capacity_ + direct_alignment - 1) / direct_alignment * direct_alignment;
        void * buffer = nullptr;
        const int allocation = ::posix_memalign(&buffer, direct_alignment, capacity_);
        if (allocation != 0) throw io_error("allocate direct-I/O buffer", allocation);
        buffer_.reset(static_cast<uint8_t *>(buffer));
        const int result = ::io_uring_queue_init(direct_queue_depth, &ring_, 0);
        if (result < 0) throw io_error("initialize snapshot io_uring", -result);
    }
    ~DirectReader() {
        // An error in one CQE must not free buffers still used by other reads.
        while (pending_ != 0) {
            io_uring_cqe * cqe = nullptr;
            const int result = ::io_uring_wait_cqe(&ring_, &cqe);
            if (result == -EINTR) continue;
            if (result < 0) break;
            ::io_uring_cqe_seen(&ring_, cqe);
            --pending_;
        }
        ::io_uring_queue_exit(&ring_);
    }
    DirectReader(const DirectReader &) = delete;
    DirectReader & operator=(const DirectReader &) = delete;

    void read_all(uint8_t * data, size_t size) {
        while (size != 0) {
            if (cursor_ == available_) refill();
            const size_t count = std::min(size, available_ - cursor_);
            std::memcpy(data, buffer_.get() + cursor_, count);
            data += count;
            size -= count;
            cursor_ += count;
        }
    }

    void finish() const {
        struct stat st{};
        if (::fstat(fd_, &st) != 0) throw io_error("check direct-I/O snapshot EOF");
        if (next_offset_ != file_size_ || cursor_ != available_ ||
            st.st_size < 0 || static_cast<uint64_t>(st.st_size) != file_size_) {
            throw std::runtime_error("snapshot length changed or trailing data");
        }
    }

private:
    void refill() {
        if (next_offset_ == file_size_) throw std::runtime_error("truncated direct-I/O snapshot");
        const size_t bytes = static_cast<size_t>(std::min<uint64_t>(capacity_, file_size_ - next_offset_));
        std::array<size_t, direct_queue_depth> expected{};
        unsigned requests = 0;
        for (size_t offset = 0; offset < bytes; offset += direct_chunk_bytes) {
            const size_t count = std::min(direct_chunk_bytes, bytes - offset);
            const size_t aligned = (count + direct_alignment - 1) / direct_alignment * direct_alignment;
            io_uring_sqe * sqe = ::io_uring_get_sqe(&ring_);
            if (!sqe) throw std::runtime_error("snapshot io_uring submission queue exhausted");
            ::io_uring_prep_read(sqe, fd_, buffer_.get() + offset, static_cast<unsigned>(aligned), next_offset_ + offset);
            ::io_uring_sqe_set_data64(sqe, requests);
            expected[requests++] = count;
        }
        unsigned submitted = 0;
        while (submitted < requests) {
            const int result = ::io_uring_submit(&ring_);
            if (result == -EINTR) continue;
            if (result < 0) throw io_error("submit snapshot io_uring reads", -result);
            if (result == 0) throw std::runtime_error("snapshot io_uring submission made no progress");
            submitted += static_cast<unsigned>(result);
            pending_ += static_cast<unsigned>(result);
        }
        std::array<bool, direct_queue_depth> completed{};
        while (pending_ != 0) {
            io_uring_cqe * cqe = nullptr;
            const int result = ::io_uring_wait_cqe(&ring_, &cqe);
            if (result == -EINTR) continue;
            if (result < 0) throw io_error("wait for snapshot io_uring read", -result);
            const uint64_t index = ::io_uring_cqe_get_data64(cqe);
            const int count = cqe->res;
            ::io_uring_cqe_seen(&ring_, cqe);
            --pending_;
            if (count < 0) throw io_error("io_uring O_DIRECT snapshot read", -count);
            if (index >= requests || completed[index]) throw std::runtime_error("unexpected snapshot io_uring completion");
            // Only the final aligned read may be short, by exactly its EOF padding.
            if (static_cast<size_t>(count) != expected[index]) throw std::runtime_error("short or changed direct-I/O snapshot");
            completed[index] = true;
        }
        next_offset_ += bytes;
        available_ = bytes;
        cursor_ = 0;
    }

    int fd_;
    uint64_t file_size_;
    uint64_t next_offset_ = 0;
    size_t capacity_ = 0;
    size_t available_ = 0;
    size_t cursor_ = 0;
    unsigned pending_ = 0;
    std::unique_ptr<uint8_t, decltype(&std::free)> buffer_{nullptr, &std::free};
    io_uring ring_{};
};

inline void write_all(int fd, const uint8_t * data, size_t size) {
    while (size != 0) {
        const ssize_t n = ::write(fd, data, std::min<size_t>(size, 1U << 20));
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) throw io_error("write buffered-file snapshot");
        if (n == 0) throw std::runtime_error("zero-length snapshot write");
        data += n;
        size -= static_cast<size_t>(n);
    }
}

inline uint64_t used_bytes(int root) {
    // A fresh directory description avoids sharing/reusing a readdir offset.
    Fd scan(::openat(root, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC));
    if (scan.get() < 0) throw io_error("open buffered-file quota scan");
    DIR * raw = ::fdopendir(scan.get());
    if (!raw) throw io_error("open buffered-file directory stream");
    scan.release();
    struct CloseDir { void operator()(DIR * dir) const { ::closedir(dir); } };
    std::unique_ptr<DIR, CloseDir> directory(raw);
    uint64_t total = 0;
    while (true) {
        errno = 0;
        const dirent * entry = ::readdir(directory.get());
        if (!entry) {
            if (errno != 0) throw io_error("scan buffered-file quota");
            break;
        }
        if (std::strcmp(entry->d_name, ".") == 0 || std::strcmp(entry->d_name, "..") == 0) continue;
        struct stat st{};
        if (::fstatat(root, entry->d_name, &st, AT_SYMLINK_NOFOLLOW) != 0) throw io_error("stat quota entry");
        check_file(st);
        const uint64_t size = static_cast<uint64_t>(st.st_size);
        if (size > std::numeric_limits<uint64_t>::max() - total) throw std::length_error("store size overflow");
        total += size; // Includes unfinished temporary files; never remove someone else's files.
    }
    return total;
}

inline void check_space(int root, uint64_t bytes, const Limits & limits) {
    struct statvfs fs{};
    if (::fstatvfs(root, &fs) != 0) throw io_error("check buffered-file free space");
    const uint64_t block = fs.f_frsize ? fs.f_frsize : fs.f_bsize;
    if (block == 0) throw std::runtime_error("invalid filesystem block size");
    const uint64_t maximum = std::numeric_limits<uint64_t>::max();
    const uint64_t available = fs.f_bavail > maximum / block ? maximum : fs.f_bavail * block;
    const uint64_t padding = (block - bytes % block) % block;
    if (bytes > maximum - padding) throw std::length_error("snapshot disk size overflow");
    const uint64_t needed = bytes + padding;
    if (available < limits.disk_reserve_bytes || needed > available - limits.disk_reserve_bytes) {
        throw std::length_error("buffered-file disk reserve would be violated");
    }
}

class TempFile {
public:
    explicit TempFile(int root) : root_(root) {
        static std::atomic<uint64_t> sequence{0};
        for (int attempt = 0; attempt < 128; ++attempt) {
            name_ = ".qcache-tmp-" + std::to_string(::getpid()) + "-" + std::to_string(sequence.fetch_add(1));
            const int fd = ::openat(root_, name_.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
            if (fd >= 0) {
                fd_ = Fd(fd);
                return;
            }
            if (errno != EEXIST) throw io_error("create exclusive snapshot staging file");
        }
        throw std::runtime_error("could not create unique snapshot staging file");
    }
    ~TempFile() { if (!name_.empty()) ::unlinkat(root_, name_.c_str(), 0); }
    TempFile(const TempFile &) = delete;
    TempFile & operator=(const TempFile &) = delete;
    int fd() const { return fd_.get(); }
    const std::string & name() const { return name_; }
    void publish(const std::string & destination) {
        // Linux no-replace rename publishes one complete inode with one link.
        // Fail closed if the filesystem lacks this operation; never fall back
        // to an overwriting rename or expose a transient second hardlink.
        if (::syscall(SYS_renameat2, root_, name_.c_str(), root_, destination.c_str(), RENAME_NOREPLACE) != 0) {
            throw io_error("publish snapshot without replacement");
        }
        name_.clear();
    }

private:
    int root_;
    std::string name_;
    Fd fd_;
};

inline std::array<uint8_t, header_bytes> make_header(const Snapshot & snapshot, uint64_t size) {
    std::array<uint8_t, header_bytes> header{};
    std::copy(magic.begin(), magic.end(), header.begin());
    put_le(header.data() + 8, 1, 4);
    put_le(header.data() + 12, header_bytes, 4);
    put_le(header.data() + 16, snapshot.identity.configuration.size(), 8);
    put_le(header.data() + 24, snapshot.identity.prefix_tokens.size(), 8);
    put_le(header.data() + 32, snapshot.identity.processed_tokens, 8);
    put_le(header.data() + 40, snapshot.state.size(), 8);
    put_le(header.data() + 48, size, 8);
    return header;
}

inline void write_record(int fd, const Snapshot & snapshot, uint64_t size) {
    const auto header = make_header(snapshot, size);
    Crc64 crc;
    crc.update(header.data(), 56);
    write_all(fd, header.data(), header.size());
    const auto write_payload = [&](const uint8_t * data, size_t count) {
        write_all(fd, data, count);
        crc.update(data, count);
    };
    write_payload(reinterpret_cast<const uint8_t *>(snapshot.identity.configuration.data()),
                  snapshot.identity.configuration.size());
    std::array<uint8_t, 4096> token_bytes{};
    for (size_t start = 0; start < snapshot.identity.prefix_tokens.size();) {
        const size_t count = std::min(token_bytes.size() / 4, snapshot.identity.prefix_tokens.size() - start);
        for (size_t i = 0; i < count; ++i) {
            put_le(token_bytes.data() + i * 4, static_cast<uint32_t>(snapshot.identity.prefix_tokens[start + i]), 4);
        }
        write_payload(token_bytes.data(), count * 4);
        start += count;
    }
    write_payload(snapshot.state.data(), snapshot.state.size());
    std::array<uint8_t, 8> checksum{};
    put_le(checksum.data(), crc.value(), checksum.size());
    // Seek only within the private staging file; it is not visible to readers yet.
    if (::lseek(fd, 56, SEEK_SET) != 56) throw io_error("seek snapshot checksum");
    write_all(fd, checksum.data(), checksum.size());
    if (::fsync(fd) != 0) throw io_error("fsync buffered-file snapshot");
}

inline LookupResult read_record(int fd, const Identity & expected, const Limits & limits) {
    struct stat st{};
    if (::fstat(fd, &st) != 0) throw io_error("stat direct-I/O snapshot");
    check_file(st);
    const uint64_t file_size = static_cast<uint64_t>(st.st_size);
    if (file_size < header_bytes) throw std::runtime_error("truncated snapshot header");
    if (file_size > limits.max_blob_bytes) throw std::length_error("snapshot exceeds max_blob_bytes");
    check_quota(0, file_size, limits);
    DirectReader reader(fd, file_size);
    std::array<uint8_t, header_bytes> header{};
    reader.read_all(header.data(), header.size());
    if (!std::equal(magic.begin(), magic.end(), header.begin())) throw std::runtime_error("invalid snapshot magic");
    if (get_le(header.data() + 8, 4) != 1) throw std::runtime_error("unsupported snapshot version");
    if (get_le(header.data() + 12, 4) != header_bytes) throw std::runtime_error("invalid snapshot header size");
    const uint64_t configuration_size = get_le(header.data() + 16, 8);
    const uint64_t token_count = get_le(header.data() + 24, 8);
    const uint64_t processed = get_le(header.data() + 32, 8);
    const uint64_t state_size = get_le(header.data() + 40, 8);
    const uint64_t size = record_size(configuration_size, token_count, processed, state_size, limits);
    check_quota(0, size, limits);
    if (get_le(header.data() + 48, 8) != size || static_cast<uint64_t>(st.st_size) != size) {
        throw std::runtime_error("snapshot length mismatch (truncated or trailing data)");
    }
    // No payload allocation occurs until every length, cap and actual file size agrees.
    auto snapshot = std::make_shared<Snapshot>();
    snapshot->identity.configuration.resize(static_cast<size_t>(configuration_size));
    snapshot->identity.prefix_tokens.resize(static_cast<size_t>(token_count));
    snapshot->identity.processed_tokens = processed;
    Crc64 crc;
    crc.update(header.data(), 56);
    auto * configuration = reinterpret_cast<uint8_t *>(snapshot->identity.configuration.data());
    reader.read_all(configuration, static_cast<size_t>(configuration_size));
    crc.update(configuration, static_cast<size_t>(configuration_size));
    std::array<uint8_t, 4096> token_bytes{};
    for (size_t start = 0; start < token_count;) {
        const size_t count = std::min(token_bytes.size() / 4, static_cast<size_t>(token_count) - start);
        reader.read_all(token_bytes.data(), count * 4);
        crc.update(token_bytes.data(), count * 4);
        for (size_t i = 0; i < count; ++i) {
            const uint64_t bits = get_le(token_bytes.data() + i * 4, 4);
            const int64_t token = bits <= INT32_MAX ? static_cast<int64_t>(bits) : static_cast<int64_t>(bits) - (1LL << 32);
            snapshot->identity.prefix_tokens[start + i] = static_cast<int32_t>(token);
        }
        start += count;
    }
    const bool compatible = snapshot->identity == expected;
    if (compatible) {
        snapshot->state.resize(static_cast<size_t>(state_size));
        reader.read_all(snapshot->state.data(), snapshot->state.size());
        crc.update(snapshot->state.data(), snapshot->state.size());
    } else {
        // Validate checksum even on identity mismatch, without allocating a huge unused state.
        std::array<uint8_t, 65536> chunk{};
        uint64_t remaining = state_size;
        while (remaining != 0) {
            const size_t count = static_cast<size_t>(std::min<uint64_t>(remaining, chunk.size()));
            reader.read_all(chunk.data(), count);
            crc.update(chunk.data(), count);
            remaining -= count;
        }
    }
    reader.finish();
    if (crc.value() != get_le(header.data() + 56, 8)) throw std::runtime_error("snapshot CRC64 mismatch");
    if (!compatible) return {LookupStatus::incompatible, nullptr, "exact configuration/token/boundary mismatch"};
    return {LookupStatus::hit, std::move(snapshot), "io-uring-direct hit (O_DIRECT; includes allocation/copy/CRC)"};
}

} // namespace detail

class MemoryStore {
public:
    explicit MemoryStore(Limits limits) : limits_(detail::valid_limits(limits)) {}

    void put(const std::string & key, Snapshot snapshot) {
        detail::valid_key(key);
        const uint64_t bytes = detail::record_size(snapshot, limits_);
        if (entries_.count(key) != 0) throw std::runtime_error("duplicate snapshot key; never overwritten");
        detail::check_quota(used_bytes_, bytes, limits_);
        entries_.emplace(key, std::make_shared<const Snapshot>(std::move(snapshot)));
        used_bytes_ += bytes;
    }

    LookupResult get(const std::string & key, const Identity & expected) const {
        detail::valid_key(key);
        const auto entry = entries_.find(key);
        if (entry == entries_.end()) return {LookupStatus::miss, nullptr, "DRAM key not found"};
        if (!(entry->second->identity == expected)) {
            return {LookupStatus::incompatible, nullptr, "exact configuration/token/boundary mismatch"};
        }
        return {LookupStatus::hit, entry->second, "DRAM hit (immutable shared state, no payload copy)"};
    }

private:
    Limits limits_;
    uint64_t used_bytes_ = 0;
    std::unordered_map<std::string, std::shared_ptr<const Snapshot>> entries_;
};

class DirectFileStore {
public:
    // Roots must be private (0700) and owned by this uid. Missing components are
    // created as 0700; existing directories/files are never chmod'ed or removed.
    // max_blob_bytes bounds the entire encoded record. max_store_bytes counts
    // encoded bytes, including incomplete staging files. Not an RSS quota.
    explicit DirectFileStore(const std::filesystem::path & root, Limits limits)
        : limits_(detail::valid_limits(limits)), root_(detail::open_root(root)) {}

    void put(const std::string & key, Snapshot snapshot) {
        detail::valid_key(key);
        const uint64_t bytes = detail::record_size(snapshot, limits_);
        const std::string name = key + ".qcache";
        struct stat existing{};
        if (::fstatat(root_.get(), name.c_str(), &existing, AT_SYMLINK_NOFOLLOW) == 0) {
            throw std::runtime_error("duplicate snapshot key; never overwritten");
        }
        if (errno != ENOENT) throw detail::io_error("check existing snapshot");
        detail::check_quota(detail::used_bytes(root_.get()), bytes, limits_);
        detail::check_space(root_.get(), bytes, limits_);
        detail::TempFile temporary(root_.get());
        detail::write_record(temporary.fd(), snapshot, bytes);
        temporary.publish(name);
        // On a durability error the complete published entry may already exist;
        // do not delete it or any other published/user file on the failure path.
        if (::fsync(root_.get()) != 0) throw detail::io_error("fsync buffered-file directory");
    }

    LookupResult get(const std::string & key, const Identity & expected) const {
        detail::valid_key(key);
        const std::string name = key + ".qcache";
        // NONBLOCK prevents a planted FIFO from blocking before fstat rejects it.
        detail::Fd fd(::openat(root_.get(), name.c_str(), O_RDONLY | O_DIRECT | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC));
        if (fd.get() < 0) {
            if (errno == ENOENT) return {LookupStatus::miss, nullptr, "io-uring-direct key not found"};
            return {LookupStatus::corrupt, nullptr, detail::io_error("open O_DIRECT snapshot (no buffered fallback)").what()};
        }
        try {
            return detail::read_record(fd.get(), expected, limits_);
        } catch (const std::exception & error) {
            return {LookupStatus::corrupt, nullptr, error.what()};
        }
    }

private:
    Limits limits_;
    detail::Fd root_;
};

inline std::vector<uint8_t> checked_export(
        size_t expected_size, const std::function<size_t(uint8_t *, size_t)> & exporter,
        const Limits & limits) {
    if (expected_size == 0 || expected_size > limits.max_blob_bytes) {
        throw std::length_error("empty or oversized exported state");
    }
    std::vector<uint8_t> state(expected_size);
    if (exporter(state.data(), state.size()) != state.size()) {
        throw std::runtime_error("state exporter returned an inexact byte count");
    }
    return state;
}

inline void checked_restore(
        const std::vector<uint8_t> & state, const std::function<void()> & clear,
        const std::function<size_t(const uint8_t *, size_t)> & importer) {
    try {
        clear();
        if (state.empty()) throw std::invalid_argument("cannot restore empty state");
        if (importer(state.data(), state.size()) != state.size()) {
            throw std::runtime_error("state importer returned an inexact byte count");
        }
    } catch (...) {
        // Try clearing even if the first clear/import threw. Preserve that error
        // if cleanup also fails; callers must stop, never continue inference.
        try { clear(); } catch (...) {}
        throw;
    }
}

struct LogitComparison {
    bool passed;
    double max_abs_error;
    double max_rel_error;
    std::string detail;
};

inline LogitComparison compare_logits(const std::vector<float> & expected,
                                     const std::vector<float> & actual,
                                     double atol = 1e-4, double rtol = 1e-4) {
    const double infinity = std::numeric_limits<double>::infinity();
    if (!std::isfinite(atol) || !std::isfinite(rtol) || atol < 0 || rtol < 0) {
        return {false, infinity, infinity, "logit tolerances must be finite and nonnegative"};
    }
    if (expected.empty() || expected.size() != actual.size()) {
        return {false, infinity, infinity, "empty or mismatched logit vectors"};
    }
    LogitComparison result{true, 0, 0, "finite logits within abs <= atol + rtol * abs(expected)"};
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(expected[i]) || !std::isfinite(actual[i])) {
            return {false, infinity, infinity, "nonfinite logit at index " + std::to_string(i)};
        }
        const double absolute = std::abs(static_cast<double>(expected[i]) - actual[i]);
        const double magnitude = std::abs(static_cast<double>(expected[i]));
        const double relative = magnitude != 0 ? absolute / magnitude : (absolute == 0 ? 0 : infinity);
        result.max_abs_error = std::max(result.max_abs_error, absolute);
        result.max_rel_error = std::max(result.max_rel_error, relative);
        if (absolute > atol + rtol * magnitude && result.passed) {
            result.passed = false;
            result.detail = "logit tolerance exceeded at index " + std::to_string(i);
        }
    }
    return result;
}

inline bool same_tokens(const std::vector<int32_t> & expected, const std::vector<int32_t> & actual) {
    return expected == actual;
}

} // namespace qwen_cache
#endif
