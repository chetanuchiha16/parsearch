#include "dsearch/search_engine.hpp"

#include "dsearch/inverted_index.hpp"
#include "dsearch/statistics.hpp"
#include "dsearch/tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace dsearch {

std::vector<SearchResult> keyword_search(
    const std::string&   query_text,
    const InvertedIndex& index,
    const Statistics&    stats,
    const Tokenizer&     tokenizer,
    int                  top_k)
{
    // ── 1. Tokenize query ─────────────────────────────────────────────────
    const auto query_tokens = tokenizer.tokenize(query_text);
    if (query_tokens.empty()) return {};

    // Count query term frequencies
    std::unordered_map<std::string, int> query_tf;
    for (const auto& t : query_tokens) ++query_tf[t];

    // ── 2. Compute query TF-IDF weights and ||q|| ─────────────────────────
    std::unordered_map<std::string, double> query_weights;
    double query_norm_sq = 0.0;

    for (const auto& [term, count] : query_tf) {
        const double idf_val = stats.idf(term);
        if (idf_val == 0.0) continue; // term not in corpus; skip

        const double tf_w = 1.0 + std::log10(static_cast<double>(count));
        const double w    = tf_w * idf_val;
        query_weights[term] = w;
        query_norm_sq += w * w;
    }

    if (query_norm_sq == 0.0) return {}; // no query terms found in corpus
    const double query_norm = std::sqrt(query_norm_sq);

    // ── 3. Accumulate dot-product scores across candidate documents ────────
    std::unordered_map<int, double> dot_products;

    for (const auto& [term, q_weight] : query_weights) {
        const PostingList* postings = index.get_postings(term);
        if (!postings) continue;

        const double idf_val = stats.idf(term);

        for (const auto& p : *postings) {
            const double tf_w   = (p.tf > 0)
                ? (1.0 + std::log10(static_cast<double>(p.tf)))
                : 0.0;
            const double d_weight = tf_w * idf_val;
            dot_products[p.doc_id] += q_weight * d_weight;
        }
    }

    // ── 4. Normalise to cosine similarity ─────────────────────────────────
    std::vector<SearchResult> results;
    results.reserve(dot_products.size());

    for (const auto& [doc_id, dot] : dot_products) {
        const DocMetadata* meta = index.get_doc_metadata(doc_id);
        if (!meta || meta->norm == 0.0) continue;

        const double score = dot / (query_norm * meta->norm);
        results.push_back({doc_id, meta->path, score});
    }

    // ── 5. Sort descending, return top-k ──────────────────────────────────
    std::sort(results.begin(), results.end(),
              [](const SearchResult& a, const SearchResult& b) {
                  return a.score > b.score;
              });

    if (static_cast<int>(results.size()) > top_k) {
        results.resize(top_k);
    }

    return results;
}

} // namespace dsearch
