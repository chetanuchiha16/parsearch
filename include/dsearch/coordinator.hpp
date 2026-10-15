#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>

#include "dsearch/document.hpp"
#include "dsearch/inverted_index.hpp"
#include "dsearch/search_engine.hpp"
#include "dsearch/statistics.hpp"

namespace dsearch {

class Tokenizer;

/// Represents a partitioned worker node in the distributed topology.
struct WorkerPartition {
    int worker_rank;
    std::vector<int> doc_ids;
};

/// Partitioning strategies for assigning documents across workers.
enum class PartitionPolicy {
    RoundRobin, // doc_id % num_workers
    Range       // Contiguous blocks
};

/// Coordinates distributed indexing and search queries across worker partitions.
/// Works with both native MPI and simulated multi-worker environments.
class DistributedCoordinator {
public:
    explicit DistributedCoordinator(int num_workers, PartitionPolicy policy = PartitionPolicy::RoundRobin);

    /// Compute partition assignments for a given total number of documents.
    [[nodiscard]] std::vector<WorkerPartition> partition_documents(int total_docs) const;

    /// Aggregate raw document frequencies from all workers to compute global Statistics.
    ///
    /// CRITICAL INVARIANT: Workers cannot average local IDF.
    /// The coordinator derives global IDF from sum of all raw DF counts:
    ///   global_df(t) = sum_{w} local_df_w(t)
    ///   global_idf(t) = log10( (global_N + 1) / (global_df(t) + 1) ) + 1
    [[nodiscard]] Statistics aggregate_global_statistics(
        int total_corpus_docs,
        const std::vector<std::unordered_map<std::string, int>>& worker_df_maps) const;

    /// Merge top-k candidate results received from distributed workers into a final ranking.
    ///
    /// Candidates are merged with deterministic tie-breaking (scores within 1e-9 broken by doc_id).
    [[nodiscard]] std::vector<SearchResult> merge_worker_results(
        const std::vector<std::vector<SearchResult>>& worker_results,
        int top_k = 10) const;

    [[nodiscard]] int num_workers() const { return num_workers_; }

private:
    int num_workers_;
    PartitionPolicy policy_;
};

/// Represents a single distributed worker owning a partition of the corpus.
class DistributedWorker {
public:
    DistributedWorker(int rank, std::vector<int> assigned_doc_ids);

    /// Index the worker's assigned documents from directory.
    void index_partition(
        const std::filesystem::path& corpus_dir,
        const Tokenizer& tokenizer);

    /// Index pre-partitioned documents directly in memory.
    void index_partition(
        const std::vector<Document>& partition_docs,
        const Tokenizer& tokenizer);

    /// Return the raw DF counts from this worker's partition.
    [[nodiscard]] const std::unordered_map<std::string, int>& get_local_df() const;

    /// Update local document norms using the globally aggregated Statistics.
    void apply_global_statistics(const Statistics& global_stats);

    /// Query the worker's local partition using the global statistics and return local top-k.
    [[nodiscard]] std::vector<SearchResult> search(
        const std::string& query,
        const Statistics& global_stats,
        const Tokenizer& tokenizer,
        int top_k) const;

    [[nodiscard]] int rank() const { return rank_; }
    [[nodiscard]] const InvertedIndex& index() const { return local_index_; }

private:
    int rank_;
    std::vector<int> assigned_doc_ids_;
    InvertedIndex local_index_;
    std::unordered_map<std::string, int> local_df_;
};

} // namespace dsearch
