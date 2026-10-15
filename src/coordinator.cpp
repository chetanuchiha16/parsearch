#include "dsearch/coordinator.hpp"
#include "dsearch/document.hpp"
#include "dsearch/inverted_index.hpp"
#include "dsearch/statistics.hpp"
#include "dsearch/tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace dsearch {

// ── DistributedCoordinator ────────────────────────────────────────────────

DistributedCoordinator::DistributedCoordinator(int num_workers, PartitionPolicy policy)
    : num_workers_(std::max(1, num_workers)), policy_(policy) {}

std::vector<WorkerPartition> DistributedCoordinator::partition_documents(int total_docs) const {
    std::vector<WorkerPartition> partitions(num_workers_);
    for (int i = 0; i < num_workers_; ++i) {
        partitions[i].worker_rank = i;
    }

    if (total_docs <= 0) return partitions;

    if (policy_ == PartitionPolicy::RoundRobin) {
        for (int doc_id = 0; doc_id < total_docs; ++doc_id) {
            int target_worker = doc_id % num_workers_;
            partitions[target_worker].doc_ids.push_back(doc_id);
        }
    } else { // PartitionPolicy::Range
        int chunk_size = total_docs / num_workers_;
        int remainder = total_docs % num_workers_;
        int current_id = 0;

        for (int i = 0; i < num_workers_; ++i) {
            int count = chunk_size + (i < remainder ? 1 : 0);
            for (int c = 0; c < count; ++c) {
                partitions[i].doc_ids.push_back(current_id++);
            }
        }
    }

    return partitions;
}

Statistics DistributedCoordinator::aggregate_global_statistics(
    int total_corpus_docs,
    const std::vector<std::unordered_map<std::string, int>>& worker_df_maps) const
{
    Statistics global_stats;
    global_stats.total_docs_ = total_corpus_docs;

    // Sum DF counts for all terms across all worker partitions
    for (const auto& w_df : worker_df_maps) {
        for (const auto& [term, count] : w_df) {
            global_stats.df_[term] += count;
        }
    }

    // Compute global IDF for each term: log10((N+1)/(df+1)) + 1
    for (const auto& [term, df_count] : global_stats.df_) {
        global_stats.idf_[term] = (total_corpus_docs > 0)
            ? (std::log10(static_cast<double>(total_corpus_docs + 1) /
                          static_cast<double>(df_count + 1)) + 1.0)
            : 0.0;
    }

    return global_stats;
}

std::vector<SearchResult> DistributedCoordinator::merge_worker_results(
    const std::vector<std::vector<SearchResult>>& worker_results,
    int top_k) const
{
    std::vector<SearchResult> merged;
    for (const auto& list : worker_results) {
        merged.insert(merged.end(), list.begin(), list.end());
    }

    // Deterministic ranking: descending score, tie-break by doc_id ascending
    std::sort(merged.begin(), merged.end(),
              [](const SearchResult& a, const SearchResult& b) {
                  if (std::abs(a.score - b.score) > 1e-9) {
                      return a.score > b.score;
                  }
                  return a.doc_id < b.doc_id;
              });

    if (static_cast<int>(merged.size()) > top_k) {
        merged.resize(top_k);
    }

    return merged;
}

// ── DistributedWorker ─────────────────────────────────────────────────────

DistributedWorker::DistributedWorker(int rank, std::vector<int> assigned_doc_ids)
    : rank_(rank), assigned_doc_ids_(std::move(assigned_doc_ids)) {}

void DistributedWorker::index_partition(
    const std::filesystem::path& corpus_dir,
    const Tokenizer& tokenizer)
{
    auto all_docs = load_documents(corpus_dir);
    std::set<int> my_ids(assigned_doc_ids_.begin(), assigned_doc_ids_.end());

    local_index_.clear();
    local_df_.clear();

    for (const auto& doc : all_docs) {
        if (my_ids.find(doc.id) != my_ids.end()) {
            auto tokens = tokenizer.tokenize(doc.content);
            local_index_.add_document(doc.id, tokens, doc.path.string());
        }
    }

    // Record local DF counts
    for (const auto& [term, postings] : local_index_.get_index()) {
        local_df_[term] = static_cast<int>(postings.size());
    }
}

const std::unordered_map<std::string, int>& DistributedWorker::get_local_df() const {
    return local_df_;
}

void DistributedWorker::apply_global_statistics(const Statistics& global_stats) {
    // Reweight local document norms using global IDF
    local_index_.compute_doc_norms(global_stats);
}

std::vector<SearchResult> DistributedWorker::search(
    const std::string& query,
    const Statistics& global_stats,
    const Tokenizer& tokenizer,
    int top_k) const
{
    return keyword_search(query, local_index_, global_stats, tokenizer, top_k);
}

} // namespace dsearch
