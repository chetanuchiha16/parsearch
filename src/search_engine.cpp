#include "dsearch/search_engine.hpp"

#include "dsearch/document.hpp"
#include "dsearch/inverted_index.hpp"
#include "dsearch/statistics.hpp"
#include "dsearch/tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <unordered_map>
#include <vector>

namespace dsearch {

int build_index_parallel(
    const std::filesystem::path& corpus_dir,
    InvertedIndex&               index,
    Statistics&                  stats,
    const Tokenizer&             tokenizer,
    int                          num_threads)
{
    auto docs = load_documents(corpus_dir);
    if (docs.empty()) return 0;

    int n_threads = num_threads > 0 ? num_threads : omp_get_max_threads();
    std::vector<InvertedIndex> thread_local_indexes(n_threads);

    const int total = static_cast<int>(docs.size());

    #pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < total; ++i) {
            auto tokens = tokenizer.tokenize(docs[i].content);
            thread_local_indexes[tid].add_document(docs[i].id, tokens, docs[i].path.string());
        }
    }

    // Merge partial thread-local indexes sequentially into the primary index
    index.clear();
    for (auto& local_idx : thread_local_indexes) {
        index.merge_from(std::move(local_idx));
    }

    // Compute stats and document norms
    stats.compute(index);
    index.compute_doc_norms(stats);
    index.sort_postings();

    return static_cast<int>(docs.size());
}

std::vector<SearchResult> keyword_search(
    const std::string&   query_text,
    const InvertedIndex& index,
    const Statistics&    stats,
    const Tokenizer&     tokenizer,
    int                  top_k,
    int                  num_threads)
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

    // ── 4. Normalise to cosine similarity (Parallel when candidates > 64) ───
    std::vector<std::pair<int, double>> candidates(dot_products.begin(), dot_products.end());
    std::vector<SearchResult> results(candidates.size());

    int n_threads = num_threads > 0 ? num_threads : omp_get_max_threads();
    const int num_candidates = static_cast<int>(candidates.size());

    #pragma omp parallel for schedule(static) num_threads(n_threads) if(num_candidates > 64)
    for (int i = 0; i < num_candidates; ++i) {
        int doc_id = candidates[i].first;
        double dot = candidates[i].second;
        const DocMetadata* meta = index.get_doc_metadata(doc_id);
        if (meta && meta->norm > 0.0) {
            double score = dot / (query_norm * meta->norm);
            results[i] = SearchResult{doc_id, meta->path, score};
        } else {
            results[i] = SearchResult{doc_id, meta ? meta->path : "", 0.0};
        }
    }

    // Filter out zero-score entries
    results.erase(std::remove_if(results.begin(), results.end(),
                                [](const SearchResult& r) { return r.score <= 0.0; }),
                  results.end());

    // ── 5. Efficient top-k selection: partial_sort in O(N log k) with tie-breaking ──
    const auto cmp = [](const SearchResult& a, const SearchResult& b) {
        if (std::abs(a.score - b.score) > 1e-9) {
            return a.score > b.score; // higher score first
        }
        return a.doc_id < b.doc_id;   // smaller doc_id first
    };

    if (static_cast<int>(results.size()) > top_k) {
        std::partial_sort(results.begin(), results.begin() + top_k, results.end(), cmp);
        results.resize(top_k);
    } else {
        std::sort(results.begin(), results.end(), cmp);
    }

    return results;
}

} // namespace dsearch
