#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace dsearch {

class Statistics; // forward declaration

// ── Posting ───────────────────────────────────────────────────────────────

/// One entry in a posting list: the document that contains the term and
/// how many times it occurs (raw term frequency).
struct Posting {
    int doc_id;
    int tf;     ///< Raw term frequency (count of occurrences)
};

using PostingList = std::vector<Posting>;

// ── DocMetadata ───────────────────────────────────────────────────────────

/// Per-document metadata stored alongside the inverted index.
struct DocMetadata {
    int         id;     ///< Unique document ID
    std::string path;   ///< Source file path
    int         length; ///< Total token count (for reference; not used in scoring)
    double      norm;   ///< L2 norm of the document's TF-IDF vector (filled by compute_doc_norms)
};

// ── InvertedIndex ─────────────────────────────────────────────────────────

/// In-memory inverted index: maps each term to the list of documents
/// that contain it and their raw term frequencies.
///
/// Usage:
///   1. Call add_document() for each document in the corpus.
///   2. Build a Statistics object from the completed index.
///   3. Call compute_doc_norms() with the Statistics to pre-compute norms.
///   4. The index is now ready for cosine-similarity search.
class InvertedIndex {
public:
    /// Index a single document.
    ///
    /// Computes term frequencies from the token list, updates the postings
    /// map, and stores document metadata. The document norm is set to 0.0
    /// until compute_doc_norms() is called.
    ///
    /// @param doc_id  Unique integer ID for the document
    /// @param tokens  Normalized terms produced by the Tokenizer
    /// @param path    Source file path stored in metadata
    void add_document(int doc_id,
                      const std::vector<std::string>& tokens,
                      const std::string& path);

    /// Merge another partial/thread-local index into this one.
    /// Used during OpenMP parallel indexing to combine thread-local partitions.
    void merge_from(InvertedIndex&& other);

    /// Return the posting list for a term, or nullptr if absent.
    [[nodiscard]] const PostingList* get_postings(const std::string& term) const;

    /// Return metadata for a document ID, or nullptr if not found.
    [[nodiscard]] const DocMetadata* get_doc_metadata(int doc_id) const;

    /// Number of documents in the index.
    [[nodiscard]] int total_docs() const { return static_cast<int>(docs_.size()); }

    /// Read-only access to the full term→postings map.
    [[nodiscard]] const std::unordered_map<std::string, PostingList>& get_index() const
    { return index_; }

    /// Read-only access to the full document metadata map.
    [[nodiscard]] const std::unordered_map<int, DocMetadata>& get_docs() const
    { return docs_; }

    /// Compute and store the L2 norm of every document's TF-IDF vector.
    ///
    /// Must be called after Statistics::compute() because it needs IDF values.
    /// norm(d) = sqrt( sum_t  (tf_weight(t,d) * idf(t))^2 )
    void compute_doc_norms(const Statistics& stats);

    /// Clear all index and document data.
    void clear() {
        index_.clear();
        docs_.clear();
    }

    friend class IndexStorage;

private:
    std::unordered_map<std::string, PostingList> index_; ///< term → postings
    std::unordered_map<int, DocMetadata>         docs_;  ///< doc_id → metadata
};

} // namespace dsearch
