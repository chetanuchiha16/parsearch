#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>

#include "dsearch/index_storage.hpp"
#include "dsearch/inverted_index.hpp"
#include "dsearch/search_engine.hpp"
#include "dsearch/statistics.hpp"
#include "dsearch/tokenizer.hpp"

using namespace dsearch;
namespace fs = std::filesystem;

class StorageTest : public ::testing::Test {
protected:
    fs::path temp_dir;

    void SetUp() override {
        temp_dir = fs::temp_directory_path() / ("dsearch_test_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()));
        fs::create_directories(temp_dir);
    }

    void TearDown() override {
        fs::remove_all(temp_dir);
    }
};

TEST_F(StorageTest, SaveAndLoadMaintainsIdenticalSearchResults) {
    Tokenizer tokenizer;
    InvertedIndex index;
    Statistics stats;

    std::vector<std::pair<int, std::string>> docs = {
        {0, "database write ahead logging recovery crash"},
        {1, "operating systems virtual memory page fault"},
        {2, "distributed systems consensus raft mpi coordinator"}
    };

    for (const auto& [id, text] : docs) {
        auto tokens = tokenizer.tokenize(text);
        index.add_document(id, tokens, "doc" + std::to_string(id) + ".txt");
    }
    stats.compute(index);
    index.compute_doc_norms(stats);

    // Baseline search results before saving
    auto baseline_results = keyword_search("database crash recovery", index, stats, tokenizer, 5);
    ASSERT_FALSE(baseline_results.empty());

    // Save index
    IndexStorage::save(temp_dir, index, stats);

    // Verify index file exists
    EXPECT_TRUE(fs::exists(temp_dir / "index.bin"));

    // Read header validation
    auto header = IndexStorage::read_header(temp_dir);
    EXPECT_EQ(header.magic, INDEX_MAGIC);
    EXPECT_EQ(header.version, INDEX_FORMAT_VERSION);
    EXPECT_EQ(header.num_docs, 3u);

    // Load into new clean structures
    InvertedIndex loaded_index;
    Statistics loaded_stats;
    IndexStorage::load(temp_dir, loaded_index, loaded_stats);

    EXPECT_EQ(loaded_index.total_docs(), 3);
    EXPECT_EQ(loaded_stats.total_docs(), 3);

    // Search on loaded index
    auto loaded_results = keyword_search("database crash recovery", loaded_index, loaded_stats, tokenizer, 5);

    // Results must be identical
    ASSERT_EQ(baseline_results.size(), loaded_results.size());
    for (size_t i = 0; i < baseline_results.size(); ++i) {
        EXPECT_EQ(baseline_results[i].doc_id, loaded_results[i].doc_id);
        EXPECT_EQ(baseline_results[i].path, loaded_results[i].path);
        EXPECT_DOUBLE_EQ(baseline_results[i].score, loaded_results[i].score);
    }
}

TEST_F(StorageTest, DetectsInvalidMagic) {
    auto index_path = temp_dir / "index.bin";
    std::ofstream out(index_path, std::ios::binary);
    uint32_t fake_magic = 0x12345678;
    out.write(reinterpret_cast<const char*>(&fake_magic), sizeof(fake_magic));
    out.close();

    InvertedIndex index;
    Statistics stats;
    EXPECT_THROW(IndexStorage::load(temp_dir, index, stats), std::runtime_error);
    EXPECT_THROW(IndexStorage::read_header(temp_dir), std::runtime_error);
}

TEST_F(StorageTest, DetectsUnsupportedVersion) {
    auto index_path = temp_dir / "index.bin";
    std::ofstream out(index_path, std::ios::binary);
    uint32_t magic = INDEX_MAGIC;
    uint32_t version = 999; // invalid/unsupported version
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.close();

    InvertedIndex index;
    Statistics stats;
    EXPECT_THROW(IndexStorage::load(temp_dir, index, stats), std::runtime_error);
    EXPECT_THROW(IndexStorage::read_header(temp_dir), std::runtime_error);
}

TEST_F(StorageTest, DetectsMissingIndexFile) {
    fs::path non_existent_dir = temp_dir / "does_not_exist";
    InvertedIndex index;
    Statistics stats;
    EXPECT_THROW(IndexStorage::load(non_existent_dir, index, stats), std::runtime_error);
    EXPECT_THROW(IndexStorage::read_header(non_existent_dir), std::runtime_error);
}
