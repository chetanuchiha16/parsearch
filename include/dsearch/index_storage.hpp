#pragma once

#include <filesystem>
#include <string>
#include <cstdint>

namespace dsearch {

class InvertedIndex;
class Statistics;

/// Binary storage format version and magic signature
inline constexpr uint32_t INDEX_MAGIC = 0x48435344; // 'DSCH' in little endian ('D','S','C','H')
inline constexpr uint32_t INDEX_FORMAT_VERSION = 1;

/// Index metadata stored in index header
struct IndexHeader {
    uint32_t magic = INDEX_MAGIC;
    uint32_t version = INDEX_FORMAT_VERSION;
    uint32_t num_docs = 0;
    uint32_t vocab_size = 0;
    uint64_t created_at = 0;
};

/// Handles serialization and deserialization of the InvertedIndex and Statistics
/// to and from a portable binary format with integrity validation.
class IndexStorage {
public:
    /// Save in-memory InvertedIndex and Statistics to a target directory.
    /// Creates the directory if it does not exist.
    ///
    /// File created: <index_dir>/index.bin
    ///
    /// @throws std::runtime_error on filesystem or I/O failure.
    static void save(const std::filesystem::path& index_dir,
                     const InvertedIndex& index,
                     const Statistics& stats);

    /// Load InvertedIndex and Statistics from a directory containing index.bin.
    ///
    /// Validates file magic, format version, and payload integrity.
    ///
    /// @throws std::runtime_error if file cannot be read, format is invalid, or corrupted.
    static void load(const std::filesystem::path& index_dir,
                     InvertedIndex& index,
                     Statistics& stats);

    /// Read only header metadata without loading full postings and documents into memory.
    static IndexHeader read_header(const std::filesystem::path& index_dir);
};

} // namespace dsearch
