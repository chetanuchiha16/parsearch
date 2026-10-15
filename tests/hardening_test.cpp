#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <vector>

#include "dsearch/coordinator.hpp"
#include "dsearch/document.hpp"
#include "dsearch/index_storage.hpp"
#include "dsearch/inverted_index.hpp"
#include "dsearch/search_engine.hpp"
#include "dsearch/statistics.hpp"
#include "dsearch/tokenizer.hpp"

using namespace dsearch;
namespace fs = std::filesystem;

class HardeningTest : public ::testing::Test {
protected:
    fs::path temp_dir;

    void SetUp() override {
        temp_dir = fs::temp_directory_path() / ("dsearch_harden_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()));
        fs::create_directories(temp_dir);
    }

    void TearDown() override {
        fs::remove_all(temp_dir);
    }
};

// 1. Handling empty files, whitespace-only files, and mixed corrupt inputs
TEST_F(HardeningTest, HandlesMalformedAndEmptyDocumentsGracefully) {
    // doc 0: empty file
    std::ofstream(temp_dir / "empty.txt").close();

    // doc 1: only whitespace and punctuation delimiters
    {
        std::ofstream f(temp_dir / "whitespace.txt");
        f << "   \n\t   !@#$%^&*()_+=-~` \t\r\n ";
    }

    // doc 2: valid document
    {
        std::ofstream f(temp_dir / "valid.txt");
        f << "A legitimate document about distributed systems recovery.\n";
    }

    // doc 3: non-text extension should be skipped by loader
    {
        std::ofstream f(temp_dir / "binary.bin");
        f << "\x00\xFF\xAA\x55";
    }

    Tokenizer tokenizer;
    InvertedIndex index;
    Statistics stats;

    int loaded = build_index_parallel(temp_dir, index, stats, tokenizer, 1);
    EXPECT_GE(loaded, 1);

    auto res = keyword_search("distributed recovery", index, stats, tokenizer, 5);
    ASSERT_EQ(res.size(), 1u);
    EXPECT_EQ(res[0].path, (temp_dir / "valid.txt").string());
}

// 2. Corrupt index file rejection (truncated header, truncated payload)
TEST_F(HardeningTest, RejectsTruncatedIndexFiles) {
    auto index_path = temp_dir / "index.bin";

    // Write only partial header (2 bytes instead of 28 bytes)
    {
        std::ofstream out(index_path, std::ios::binary);
        uint16_t partial = 0x4843;
        out.write(reinterpret_cast<const char*>(&partial), sizeof(partial));
    }

    InvertedIndex index;
    Statistics stats;
    EXPECT_THROW(IndexStorage::load(temp_dir, index, stats), std::runtime_error);
    EXPECT_THROW(IndexStorage::read_header(temp_dir), std::runtime_error);
}

// 3. Empty directory handling
TEST_F(HardeningTest, EmptyDirectoryReturnsZeroDocuments) {
    fs::path empty_dir = temp_dir / "empty_dir";
    fs::create_directories(empty_dir);

    Tokenizer tokenizer;
    InvertedIndex index;
    Statistics stats;
    int count = build_index_parallel(empty_dir, index, stats, tokenizer, 2);
    EXPECT_EQ(count, 0);
    EXPECT_EQ(index.total_docs(), 0);
    EXPECT_EQ(stats.total_docs(), 0);
}

// 4. Non-existent corpus directory raises exception
TEST_F(HardeningTest, NonExistentDirectoryThrows) {
    fs::path non_existent = temp_dir / "does_not_exist";
    Tokenizer tokenizer;
    InvertedIndex index;
    Statistics stats;
    EXPECT_THROW(build_index_parallel(non_existent, index, stats, tokenizer, 1), std::runtime_error);
}

// 5. Distributed coordinator handling of empty partitions
TEST_F(HardeningTest, DistributedCoordinatorHandlesEmptyPartitions) {
    // 0 documents distributed across 4 workers
    DistributedCoordinator coordinator(4);
    auto partitions = coordinator.partition_documents(0);
    EXPECT_EQ(partitions.size(), 4u);
    for (const auto& p : partitions) {
        EXPECT_TRUE(p.doc_ids.empty());
    }

    // Fewer documents than workers (2 docs across 4 workers)
    auto small_partitions = coordinator.partition_documents(2);
    EXPECT_EQ(small_partitions.size(), 4u);
    EXPECT_EQ(small_partitions[0].doc_ids.size(), 1u);
    EXPECT_EQ(small_partitions[1].doc_ids.size(), 1u);
    EXPECT_EQ(small_partitions[2].doc_ids.size(), 0u);
    EXPECT_EQ(small_partitions[3].doc_ids.size(), 0u);

    // Merge empty worker results
    std::vector<std::vector<SearchResult>> empty_worker_results(4);
    auto merged = coordinator.merge_worker_results(empty_worker_results, 5);
    EXPECT_TRUE(merged.empty());
}

// 6. Query with extreme/malformed strings
TEST_F(HardeningTest, QueryWithOnlyPunctuationOrDelimitersReturnsEmpty) {
    // Setup one document
    {
        std::ofstream f(temp_dir / "doc.txt");
        f << "The fast brown fox jumps over the lazy dog.\n";
    }

    Tokenizer tokenizer;
    InvertedIndex index;
    Statistics stats;
    build_index_parallel(temp_dir, index, stats, tokenizer, 1);

    EXPECT_TRUE(keyword_search("!@#$%^&*()", index, stats, tokenizer, 5).empty());
    EXPECT_TRUE(keyword_search("   \t\n   ", index, stats, tokenizer, 5).empty());
    EXPECT_TRUE(keyword_search("a b c d", index, stats, tokenizer, 5).empty()); // all filtered (<2 chars)
}
