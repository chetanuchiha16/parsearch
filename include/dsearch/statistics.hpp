#pragma once

#include <string>
#include <unordered_map>

namespace dsearch {

class InvertedIndex; // forward declaration

/// Corpus-level statistics derived from a completed InvertedIndex.
///
/// Stores document frequencies (DF) and precomputed inverse document
/// frequencies (IDF) for every term in the vocabulary.
///
/// IDF formula used throughout the engine (smoothed variant):
///
///   idf(t) = log10( (N+1) / (df(t)+1) ) + 1
///
/// The additive +1 smoothing prevents IDF from reaching 0 when a term appears
/// in every document (df == N), and ensures single-document corpora produce
/// non-zero scores.  The trailing +1 guarantees idf >= 1 for any term present
/// in the corpus.
///
/// where N = total document count and df(t) = number of documents containing t.
///
/// IMPORTANT (distributed mode, Phase 04):
///   Workers must share raw DF counts and let the coordinator derive global
///   IDF from the global N. Averaging worker-local IDF values gives wrong
///   results because idf is not a linear function of df.
class Statistics {
public:
    /// Compute DF and IDF for every term appearing in the index.
    void compute(const InvertedIndex& index);

    /// IDF for a term; returns 0.0 if the term is not in the corpus.
    [[nodiscard]] double idf(const std::string& term) const;

    /// Document frequency for a term; returns 0 if absent.
    [[nodiscard]] int df(const std::string& term) const;

    /// Total number of documents in the corpus.
    [[nodiscard]] int total_docs() const { return total_docs_; }

    /// Read-only access to the full IDF map (used by InvertedIndex::compute_doc_norms).
    [[nodiscard]] const std::unordered_map<std::string, double>& idf_map() const
    { return idf_; }

private:
    std::unordered_map<std::string, int>    df_;
    std::unordered_map<std::string, double> idf_;
    int total_docs_ = 0;
};

} // namespace dsearch
