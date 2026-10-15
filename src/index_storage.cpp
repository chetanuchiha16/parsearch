#include "dsearch/index_storage.hpp"
#include "dsearch/inverted_index.hpp"
#include "dsearch/statistics.hpp"

#include <chrono>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace dsearch {

namespace {

// Helpers for binary I/O
template <typename T>
void write_val(std::ofstream& out, const T& val) {
    out.write(reinterpret_cast<const char*>(&val), sizeof(T));
}

template <typename T>
void read_val(std::ifstream& in, T& val) {
    in.read(reinterpret_cast<char*>(&val), sizeof(T));
}

void write_str(std::ofstream& out, const std::string& str) {
    uint32_t len = static_cast<uint32_t>(str.size());
    write_val(out, len);
    if (len > 0) {
        out.write(str.data(), len);
    }
}

std::string read_str(std::ifstream& in) {
    uint32_t len = 0;
    read_val(in, len);
    if (len == 0) return {};
    std::string str(len, '\0');
    in.read(&str[0], len);
    return str;
}

} // namespace

void IndexStorage::save(const std::filesystem::path& index_dir,
                        const InvertedIndex& index,
                        const Statistics& stats) {
    std::filesystem::create_directories(index_dir);
    auto index_path = index_dir / "index.bin";

    std::ofstream out(index_path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open index file for writing: " + index_path.string());
    }

    // 1. Header
    IndexHeader header;
    header.magic = INDEX_MAGIC;
    header.version = INDEX_FORMAT_VERSION;
    header.num_docs = static_cast<uint32_t>(index.total_docs());
    header.vocab_size = static_cast<uint32_t>(index.get_index().size());
    header.created_at = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());

    write_val(out, header.magic);
    write_val(out, header.version);
    write_val(out, header.num_docs);
    write_val(out, header.vocab_size);
    write_val(out, header.created_at);

    // 2. Documents metadata
    // Number of docs
    write_val(out, header.num_docs);
    for (const auto& [id, meta] : index.get_docs()) {
        write_val(out, meta.id);
        write_str(out, meta.path);
        write_val(out, meta.length);
        write_val(out, meta.norm);
    }

    // 3. Inverted index (postings)
    // Vocab size
    write_val(out, header.vocab_size);
    for (const auto& [term, postings] : index.get_index()) {
        write_str(out, term);
        uint32_t num_postings = static_cast<uint32_t>(postings.size());
        write_val(out, num_postings);
        // Ensure postings written to disk are sorted by doc_id ascending
        auto sorted_postings = postings;
        std::sort(sorted_postings.begin(), sorted_postings.end(),
                  [](const Posting& a, const Posting& b) {
                      return a.doc_id < b.doc_id;
                  });
        for (const auto& p : sorted_postings) {
            write_val(out, p.doc_id);
            write_val(out, p.tf);
        }
    }

    // 4. Statistics (corpus stats: DF & IDF)
    uint32_t total_docs = static_cast<uint32_t>(stats.total_docs());
    write_val(out, total_docs);

    uint32_t stats_df_size = static_cast<uint32_t>(stats.df_map().size());
    write_val(out, stats_df_size);
    for (const auto& [term, df_val] : stats.df_map()) {
        write_str(out, term);
        write_val(out, df_val);
    }

    uint32_t stats_idf_size = static_cast<uint32_t>(stats.idf_map().size());
    write_val(out, stats_idf_size);
    for (const auto& [term, idf_val] : stats.idf_map()) {
        write_str(out, term);
        write_val(out, idf_val);
    }

    if (!out.good()) {
        throw std::runtime_error("Write error occurred while serializing index to: " + index_path.string());
    }
}

IndexHeader IndexStorage::read_header(const std::filesystem::path& index_dir) {
    auto index_path = index_dir / "index.bin";
    std::ifstream in(index_path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open index file: " + index_path.string());
    }

    IndexHeader header;
    read_val(in, header.magic);
    read_val(in, header.version);
    read_val(in, header.num_docs);
    read_val(in, header.vocab_size);
    read_val(in, header.created_at);

    if (!in.good()) {
        throw std::runtime_error("Corrupted header or premature EOF in: " + index_path.string());
    }

    if (header.magic != INDEX_MAGIC) {
        throw std::runtime_error("Invalid index file magic identifier (expected 'DSCH').");
    }

    if (header.version != INDEX_FORMAT_VERSION) {
        throw std::runtime_error("Unsupported index format version: " + std::to_string(header.version));
    }

    return header;
}

void IndexStorage::load(const std::filesystem::path& index_dir,
                        InvertedIndex& index,
                        Statistics& stats) {
    auto index_path = index_dir / "index.bin";
    std::ifstream in(index_path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open index file: " + index_path.string());
    }

    // Read and validate header
    IndexHeader header;
    read_val(in, header.magic);
    read_val(in, header.version);
    read_val(in, header.num_docs);
    read_val(in, header.vocab_size);
    read_val(in, header.created_at);

    if (header.magic != INDEX_MAGIC) {
        throw std::runtime_error("Invalid index file magic identifier in: " + index_path.string());
    }
    if (header.version != INDEX_FORMAT_VERSION) {
        throw std::runtime_error("Unsupported index format version: " + std::to_string(header.version));
    }

    index.clear();
    stats.clear();

    // 2. Documents metadata
    uint32_t num_docs = 0;
    read_val(in, num_docs);
    for (uint32_t i = 0; i < num_docs; ++i) {
        DocMetadata meta;
        read_val(in, meta.id);
        meta.path = read_str(in);
        read_val(in, meta.length);
        read_val(in, meta.norm);
        index.docs_[meta.id] = std::move(meta);
    }

    // 3. Postings
    uint32_t vocab_size = 0;
    read_val(in, vocab_size);
    for (uint32_t i = 0; i < vocab_size; ++i) {
        std::string term = read_str(in);
        uint32_t num_postings = 0;
        read_val(in, num_postings);
        PostingList postings;
        postings.resize(num_postings);
        for (uint32_t j = 0; j < num_postings; ++j) {
            read_val(in, postings[j].doc_id);
            read_val(in, postings[j].tf);
        }
        index.index_[std::move(term)] = std::move(postings);
    }

    // 4. Statistics
    uint32_t total_docs = 0;
    read_val(in, total_docs);
    stats.total_docs_ = static_cast<int>(total_docs);

    uint32_t stats_df_size = 0;
    read_val(in, stats_df_size);
    for (uint32_t i = 0; i < stats_df_size; ++i) {
        std::string term = read_str(in);
        int df_val = 0;
        read_val(in, df_val);
        stats.df_[std::move(term)] = df_val;
    }

    uint32_t stats_idf_size = 0;
    read_val(in, stats_idf_size);
    for (uint32_t i = 0; i < stats_idf_size; ++i) {
        std::string term = read_str(in);
        double idf_val = 0.0;
        read_val(in, idf_val);
        stats.idf_[std::move(term)] = idf_val;
    }

    if (!in.good() && !in.eof()) {
        throw std::runtime_error("Corrupted index file or read error: " + index_path.string());
    }
}

} // namespace dsearch
