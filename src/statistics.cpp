#include "dsearch/statistics.hpp"
#include "dsearch/inverted_index.hpp"

#include <cmath>

namespace dsearch {

void Statistics::compute(const InvertedIndex& index) {
    total_docs_ = index.total_docs();
    df_.clear();
    idf_.clear();

    for (const auto& [term, postings] : index.get_index()) {
        const int df_count = static_cast<int>(postings.size());
        df_[term] = df_count;

        // Smoothed IDF: idf(t) = log10( (N+1) / (df(t)+1) ) + 1
        //
        // The +1 additive smoothing prevents idf from reaching 0 when a term
        // appears in every document (df == N), and handles the single-document
        // corpus case (N == 1) where naive idf would be log10(1) == 0 for every
        // term.  The trailing +1 ensures idf >= 1 so TF-IDF weights are always
        // positive for terms that appear in the corpus.
        idf_[term] = (total_docs_ > 0)
                         ? (std::log10(static_cast<double>(total_docs_ + 1) /
                                       static_cast<double>(df_count + 1)) + 1.0)
                         : 0.0;
    }
}

double Statistics::idf(const std::string& term) const {
    const auto it = idf_.find(term);
    return (it != idf_.end()) ? it->second : 0.0;
}

int Statistics::df(const std::string& term) const {
    const auto it = df_.find(term);
    return (it != df_.end()) ? it->second : 0;
}

} // namespace dsearch
