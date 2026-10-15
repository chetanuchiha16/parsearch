#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <cmath>

#include "dsearch/document.hpp"
#include "dsearch/inverted_index.hpp"
#include "dsearch/search_engine.hpp"
#include "dsearch/statistics.hpp"
#include "dsearch/tokenizer.hpp"

using namespace dsearch;
namespace fs = std::filesystem;

class ParallelTest : public ::testing::Test {
protected:
    fs::path temp_dir;

    void SetUp() override {
        temp_dir = fs::temp_directory_path() / ("dsearch_par_test_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()));
        fs::create_directories(temp_dir);

        // Create a test corpus of 20 documents
        for (int i = 0; i < 20; ++i) {
            std::ofstream f(temp_dir / ("doc_" + std::to_string(i) + ".txt"));
            f << "Document " << i << " discusses database crash recovery and transaction write ahead logging.\n";
            if (i % 2 == 0) {
                f << "Also includes operating system paging and virtual memory concepts.\n";
            }
            if (i % 3 == 0) {
                f << "Distributed consensus algorithms such as Paxos and Raft coordinate nodes.\n";
            }
            f.close();
        }
    }

    void TearDown() override {
        fs::remove_all(temp_dir);
    }
};

TEST_F(ParallelTest, ParallelIndexingMatchesSequentialResults) {
    Tokenizer tokenizer;

    // 1. Build sequential index (1 thread)
    InvertedIndex seq_index;
    Statistics seq_stats;
    int seq_count = build_index_parallel(temp_dir, seq_index, seq_stats, tokenizer, 1);
    EXPECT_EQ(seq_count, 20);

    // 2. Build parallel index (4 threads)
    InvertedIndex par_index;
    Statistics par_stats;
    int par_count = build_index_parallel(temp_dir, par_index, par_stats, tokenizer, 4);
    EXPECT_EQ(par_count, 20);

    // Document counts and vocab size must match
    EXPECT_EQ(seq_index.total_docs(), par_index.total_docs());
    EXPECT_EQ(seq_index.get_index().size(), par_index.get_index().size());
    EXPECT_EQ(seq_stats.total_docs(), par_stats.total_docs());

    // Search query on both indexes
    std::string query = "database crash recovery consensus";
    auto seq_res = keyword_search(query, seq_index, seq_stats, tokenizer, 10, 1);
    auto par_res = keyword_search(query, par_index, par_stats, tokenizer, 10, 4);

    ASSERT_EQ(seq_res.size(), par_res.size());
    for (size_t i = 0; i < seq_res.size(); ++i) {
        EXPECT_EQ(seq_res[i].doc_id, par_res[i].doc_id);
        EXPECT_NEAR(seq_res[i].score, par_res[i].score, 1e-9);
    }
}

TEST_F(ParallelTest, ScoringScalesSafelyAcrossThreadCounts) {
    Tokenizer tokenizer;
    InvertedIndex index;
    Statistics stats;
    build_index_parallel(temp_dir, index, stats, tokenizer, 2);

    std::string query = "operating system paging memory";
    auto res_1t = keyword_search(query, index, stats, tokenizer, 5, 1);
    auto res_2t = keyword_search(query, index, stats, tokenizer, 5, 2);
    auto res_4t = keyword_search(query, index, stats, tokenizer, 5, 4);

    ASSERT_EQ(res_1t.size(), res_2t.size());
    ASSERT_EQ(res_1t.size(), res_4t.size());

    for (size_t i = 0; i < res_1t.size(); ++i) {
        EXPECT_EQ(res_1t[i].doc_id, res_2t[i].doc_id);
        EXPECT_EQ(res_1t[i].doc_id, res_4t[i].doc_id);
        EXPECT_NEAR(res_1t[i].score, res_2t[i].score, 1e-9);
        EXPECT_NEAR(res_1t[i].score, res_4t[i].score, 1e-9);
    }
}
