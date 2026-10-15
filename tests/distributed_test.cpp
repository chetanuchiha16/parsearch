#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <cmath>

#include "dsearch/coordinator.hpp"
#include "dsearch/document.hpp"
#include "dsearch/inverted_index.hpp"
#include "dsearch/search_engine.hpp"
#include "dsearch/statistics.hpp"
#include "dsearch/tokenizer.hpp"

using namespace dsearch;
namespace fs = std::filesystem;

class DistributedTest : public ::testing::Test {
protected:
    fs::path temp_dir;

    void SetUp() override {
        temp_dir = fs::temp_directory_path() / ("dsearch_dist_test_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()));
        fs::create_directories(temp_dir);

        // Create 24 distinct documents
        for (int i = 0; i < 24; ++i) {
            std::ofstream f(temp_dir / ("doc_" + std::to_string(i) + ".txt"));
            f << "Document " << i << " covers database write ahead logging and crash recovery.\n";
            if (i % 2 == 0) f << "Contains details about operating system virtual memory page replacement.\n";
            if (i % 3 == 0) f << "Discusses distributed consensus protocols like raft and paxos.\n";
            if (i % 5 == 0) f << "Addresses network protocols tcp and socket multiplexing.\n";
            f.close();
        }
    }

    void TearDown() override {
        fs::remove_all(temp_dir);
    }
};

TEST_F(DistributedTest, DeterministicPartitioningCoversAllDocuments) {
    DistributedCoordinator coordinator(4, PartitionPolicy::RoundRobin);
    auto partitions = coordinator.partition_documents(24);

    EXPECT_EQ(partitions.size(), 4u);
    int total_assigned = 0;
    std::set<int> seen_ids;

    for (const auto& part : partitions) {
        EXPECT_EQ(part.doc_ids.size(), 6u);
        for (int id : part.doc_ids) {
            seen_ids.insert(id);
            ++total_assigned;
        }
    }

    EXPECT_EQ(total_assigned, 24);
    EXPECT_EQ(seen_ids.size(), 24u);
}

TEST_F(DistributedTest, DistributedMatchesSequentialBaselineWithinNumericalTolerance) {
    Tokenizer tokenizer;

    // 1. Compute sequential baseline
    InvertedIndex seq_index;
    Statistics seq_stats;
    int seq_docs = build_index_parallel(temp_dir, seq_index, seq_stats, tokenizer, 1);
    EXPECT_EQ(seq_docs, 24);

    std::string query = "database crash recovery raft consensus";
    auto seq_results = keyword_search(query, seq_index, seq_stats, tokenizer, 10, 1);
    ASSERT_FALSE(seq_results.empty());

    // 2. Run across 4 distributed workers
    const int num_workers = 4;
    DistributedCoordinator coordinator(num_workers);
    auto partitions = coordinator.partition_documents(24);

    std::vector<DistributedWorker> workers;
    std::vector<std::unordered_map<std::string, int>> worker_dfs;

    for (const auto& part : partitions) {
        DistributedWorker w(part.worker_rank, part.doc_ids);
        w.index_partition(temp_dir, tokenizer);
        worker_dfs.push_back(w.get_local_df());
        workers.push_back(std::move(w));
    }

    // Coordinator computes global stats
    Statistics global_stats = coordinator.aggregate_global_statistics(24, worker_dfs);

    // Verify global stats match sequential stats exactly
    EXPECT_EQ(global_stats.total_docs(), seq_stats.total_docs());
    for (const auto& [term, df_count] : seq_stats.df_map()) {
        EXPECT_EQ(global_stats.df(term), df_count);
        EXPECT_DOUBLE_EQ(global_stats.idf(term), seq_stats.idf(term));
    }

    // Workers apply global stats
    for (auto& w : workers) {
        w.apply_global_statistics(global_stats);
    }

    // Dispatch query to all workers and collect results
    std::vector<std::vector<SearchResult>> worker_results;
    for (const auto& w : workers) {
        worker_results.push_back(w.search(query, global_stats, tokenizer, 10));
    }

    // Merge at coordinator
    auto dist_results = coordinator.merge_worker_results(worker_results, 10);

    // Assert that distributed results match sequential baseline within 1e-6
    ASSERT_EQ(seq_results.size(), dist_results.size());
    for (size_t i = 0; i < seq_results.size(); ++i) {
        EXPECT_EQ(seq_results[i].doc_id, dist_results[i].doc_id);
        EXPECT_EQ(seq_results[i].path, dist_results[i].path);
        EXPECT_NEAR(seq_results[i].score, dist_results[i].score, 1e-6);
    }
}

TEST_F(DistributedTest, DistributedMatchesSequentialWithTwoWorkers) {
    Tokenizer tokenizer;

    InvertedIndex seq_index;
    Statistics seq_stats;
    build_index_parallel(temp_dir, seq_index, seq_stats, tokenizer, 1);

    std::string query = "virtual memory page replacement network";
    auto seq_results = keyword_search(query, seq_index, seq_stats, tokenizer, 8, 1);
    ASSERT_FALSE(seq_results.empty());

    const int num_workers = 2;
    DistributedCoordinator coordinator(num_workers);
    auto partitions = coordinator.partition_documents(24);

    std::vector<DistributedWorker> workers;
    std::vector<std::unordered_map<std::string, int>> worker_dfs;

    for (const auto& part : partitions) {
        DistributedWorker w(part.worker_rank, part.doc_ids);
        w.index_partition(temp_dir, tokenizer);
        worker_dfs.push_back(w.get_local_df());
        workers.push_back(std::move(w));
    }

    Statistics global_stats = coordinator.aggregate_global_statistics(24, worker_dfs);

    for (auto& w : workers) {
        w.apply_global_statistics(global_stats);
    }

    std::vector<std::vector<SearchResult>> worker_results;
    for (const auto& w : workers) {
        worker_results.push_back(w.search(query, global_stats, tokenizer, 8));
    }

    auto dist_results = coordinator.merge_worker_results(worker_results, 8);

    ASSERT_EQ(seq_results.size(), dist_results.size());
    for (size_t i = 0; i < seq_results.size(); ++i) {
        EXPECT_EQ(seq_results[i].doc_id, dist_results[i].doc_id);
        EXPECT_NEAR(seq_results[i].score, dist_results[i].score, 1e-6);
    }
}
