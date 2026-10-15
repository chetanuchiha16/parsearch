#pragma once

#include <string>
#include <vector>

namespace dsearch {

class InvertedIndex;
class Statistics;
class Tokenizer;

/// A single ranked result returned by a search query.
struct SearchResult {
    int         doc_id;  ///< Document ID within the index
    std::string path;    ///< Source file path
    double      score;   ///< Cosine similarity score in [0, 1]
};

/// Execute a TF-IDF keyword search over a built index.
///
/// Algorithm (sparse cosine similarity):
///   1. Tokenize the query with the same Tokenizer used at index time.
///   2. Compute query TF-IDF weights:  q_w(t) = tf_weight(t,q) * idf(t)
///   3. For each query term, walk its posting list and accumulate:
///         dot(q, d)  +=  q_w(t) * d_w(t,d)
///      where  d_w(t,d) = tf_weight(t,d) * idf(t)
///   4. Normalise:  score(q, d) = dot(q, d) / ( ||q|| * ||d|| )
///   5. Sort by score descending and return the top-k slice.
///
/// TF weighting (log-normalised):  tf_weight(t, x) = 1 + log10( count(t, x) )
///
/// @param query_text   Raw query string (will be tokenized internally)
/// @param index        Built inverted index (norms must be computed)
/// @param stats        Corpus statistics (IDF values)
/// @param tokenizer    Tokenizer used at index time (must be identical)
/// @param top_k        Maximum number of results to return
/// @returns            Ranked results (may be fewer than top_k)
std::vector<SearchResult> keyword_search(
    const std::string&   query_text,
    const InvertedIndex& index,
    const Statistics&    stats,
    const Tokenizer&     tokenizer,
    int                  top_k = 10
);

} // namespace dsearch
