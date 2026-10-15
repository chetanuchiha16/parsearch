#include "dsearch/inverted_index.hpp"
#include "dsearch/statistics.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace dsearch {

void InvertedIndex::add_document(int doc_id,
                                  const std::vector<std::string>& tokens,
                                  const std::string& path) {
    // Count raw term frequencies for this document
    std::unordered_map<std::string, int> tf_counts;
    for (const auto& token : tokens) {
        ++tf_counts[token];
    }

    // Insert or update posting lists
    for (const auto& [term, count] : tf_counts) {
        index_[term].push_back({doc_id, count});
    }

    // Record document metadata; norm is computed later via compute_doc_norms()
    docs_[doc_id] = DocMetadata{
        doc_id,
        path,
        static_cast<int>(tokens.size()),
        0.0  // placeholder; filled by compute_doc_norms()
    };
}

void InvertedIndex::merge_from(InvertedIndex&& other) {
    // Move or append postings
    for (auto& [term, other_postings] : other.index_) {
        auto& this_postings = index_[term];
        if (this_postings.empty()) {
            this_postings = std::move(other_postings);
        } else {
            this_postings.insert(this_postings.end(),
                                 std::make_move_iterator(other_postings.begin()),
                                 std::make_move_iterator(other_postings.end()));
        }
    }

    // Merge document metadata
    for (auto& [doc_id, meta] : other.docs_) {
        docs_[doc_id] = std::move(meta);
    }

    other.index_.clear();
    other.docs_.clear();
}

const PostingList* InvertedIndex::get_postings(const std::string& term) const {
    const auto it = index_.find(term);
    return (it != index_.end()) ? &it->second : nullptr;
}

const DocMetadata* InvertedIndex::get_doc_metadata(int doc_id) const {
    const auto it = docs_.find(doc_id);
    return (it != docs_.end()) ? &it->second : nullptr;
}

void InvertedIndex::compute_doc_norms(const Statistics& stats) {
    // Accumulate squared TF-IDF weights per document across all terms
    std::unordered_map<int, double> squared_sums;

    for (const auto& [term, postings] : index_) {
        const double idf_val = stats.idf(term);
        if (idf_val == 0.0) continue;

        for (const auto& p : postings) {
            // Log-normalised TF: 1 + log10(count)
            const double tf_w = (p.tf > 0)
                ? (1.0 + std::log10(static_cast<double>(p.tf)))
                : 0.0;
            const double w = tf_w * idf_val;
            squared_sums[p.doc_id] += w * w;
        }
    }

    // Store the L2 norm for each document
    for (auto& [doc_id, meta] : docs_) {
        const auto it = squared_sums.find(doc_id);
        meta.norm = (it != squared_sums.end()) ? std::sqrt(it->second) : 0.0;
    }
}

void InvertedIndex::sort_postings() {
    for (auto& [term, postings] : index_) {
        std::sort(postings.begin(), postings.end(),
                  [](const Posting& a, const Posting& b) {
                      return a.doc_id < b.doc_id;
                  });
    }
}

} // namespace dsearch
