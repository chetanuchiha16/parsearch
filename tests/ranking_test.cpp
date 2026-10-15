#include <gtest/gtest.h>
#include <cmath>

#include "dsearch/inverted_index.hpp"
#include "dsearch/search_engine.hpp"
#include "dsearch/statistics.hpp"
#include "dsearch/tokenizer.hpp"

using namespace dsearch;

// ── Test fixture helper ───────────────────────────────────────────────────

/// Builds a small index from a list of (id, text) pairs.
struct TestCorpus {
    InvertedIndex index;
    Statistics    stats;
    Tokenizer     tokenizer;

    explicit TestCorpus(const std::vector<std::pair<int, std::string>>& docs) {
        for (const auto& [id, text] : docs) {
            auto tokens = tokenizer.tokenize(text);
            index.add_document(id, tokens, "doc" + std::to_string(id) + ".txt");
        }
        stats.compute(index);
        index.compute_doc_norms(stats);
    }

    std::vector<SearchResult> search(const std::string& query, int top_k = 10) const {
        return keyword_search(query, index, stats, tokenizer, top_k);
    }
};

// ── Correctness tests ─────────────────────────────────────────────────────

TEST(Ranking, QueryTermDocumentIsReturned) {
    TestCorpus tc({
        {0, "database recovery crash"},
        {1, "file system operations"},
        {2, "network protocol design"},
    });
    auto results = tc.search("database");
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].doc_id, 0);
}

TEST(Ranking, NonMatchingDocumentAbsent) {
    TestCorpus tc({
        {0, "cat sat mat"},
        {1, "dog bit man"},
        {2, "cat ate rat"},
    });
    auto results = tc.search("cat");
    for (const auto& r : results) {
        EXPECT_NE(r.doc_id, 1) << "'dog bit man' must not match 'cat'";
    }
}

TEST(Ranking, MoreMatchingTermsRanksHigher) {
    // doc 0 contains both query terms; doc 1 contains only one
    TestCorpus tc({
        {0, "cat sat mat"},    // both "cat" and "sat"
        {1, "cat ate rat"},    // only "cat"
        {2, "dog bit man"},    // neither
    });
    auto results = tc.search("cat sat");
    ASSERT_GE(results.size(), 2u);
    EXPECT_EQ(results[0].doc_id, 0) << "doc with both terms must rank first";
}

TEST(Ranking, HigherTFRanksHigher) {
    // doc 0: "cat" appears twice → higher TF-IDF weight → higher score
    // doc 1: "cat" appears once
    TestCorpus tc({
        {0, "cat cat dog"},
        {1, "cat bird"},
        {2, "dog bird"},
    });
    auto results = tc.search("cat");
    ASSERT_GE(results.size(), 2u);
    EXPECT_EQ(results[0].doc_id, 0) << "tf=2 must rank above tf=1";
    EXPECT_GT(results[0].score, results[1].score);
}

TEST(Ranking, ScoresInDescendingOrder) {
    TestCorpus tc({
        {0, "database recovery crash journal"},
        {1, "database management system"},
        {2, "file system operations"},
    });
    auto results = tc.search("database recovery");
    for (size_t i = 1; i < results.size(); ++i) {
        EXPECT_GE(results[i - 1].score, results[i].score)
            << "Results must be in descending score order";
    }
}

TEST(Ranking, EmptyQueryReturnsEmpty) {
    TestCorpus tc({{0, "hello world"}, {1, "foo bar baz"}});
    EXPECT_TRUE(tc.search("").empty());
}

TEST(Ranking, TermAbsentFromCorpusReturnsEmpty) {
    TestCorpus tc({{0, "cat sat mat"}, {1, "dog bit man"}});
    EXPECT_TRUE(tc.search("quantum").empty());
}

TEST(Ranking, TopKLimitsResultCount) {
    TestCorpus tc({
        {0, "the quick brown fox jumps"},
        {1, "the lazy dog sat"},
        {2, "the cat and the hat"},
    });
    // All three contain "the"; ask for only 2
    auto results = tc.search("the", 2);
    EXPECT_LE(results.size(), 2u);
}

TEST(Ranking, SingleDocumentCorpus) {
    TestCorpus tc({{0, "hello world search engine"}});
    auto results = tc.search("search");
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].doc_id, 0);
    EXPECT_GT(results[0].score, 0.0);
    EXPECT_LE(results[0].score, 1.0 + 1e-9); // cosine similarity in [0,1]
}

TEST(Ranking, AllQueryTermsMissingReturnsEmpty) {
    TestCorpus tc({{0, "cat sat"}, {1, "dog bit"}});
    // Terms not in index at all
    EXPECT_TRUE(tc.search("xyz abc").empty());
}

// ── Manual TF-IDF verification ────────────────────────────────────────────
//
// Corpus:
//   doc0: "cat sat mat"   tokens: cat, sat, mat
//   doc1: "cat ate rat"   tokens: cat, ate, rat
//   doc2: "dog bit man"   tokens: dog, bit, man
//
// N=3, df("cat")=2, idf("cat") = log10(3/2) ≈ 0.17609
// All other terms unique: df=1, idf = log10(3) ≈ 0.47712
//
// Query: "cat sat"
//   q_tf("cat")=1, q_tf("sat")=1
//   q_w("cat") = (1+log10(1)) * 0.17609 = 0.17609
//   q_w("sat") = (1+log10(1)) * 0.47712 = 0.47712
//   ||q|| = sqrt(0.17609^2 + 0.47712^2) ≈ 0.50849
//
// doc0 dot product:
//   cat: 0.17609 * 0.17609 = 0.03100
//   sat: 0.47712 * 0.47712 = 0.22764
//   total dot = 0.25864
// doc0 norm = sqrt(0.17609^2 + 0.47712^2 + 0.47712^2) ≈ 0.69966
// score(doc0) = 0.25864 / (0.50849 * 0.69966) ≈ 0.72734
//
// doc1 dot product:
//   cat: 0.17609 * 0.17609 = 0.03100
//   sat: not in doc1 → 0
//   total dot = 0.03100
// doc1 norm = sqrt(0.17609^2 + 0.47712^2 + 0.47712^2) ≈ 0.69966 (same structure)
// score(doc1) = 0.03100 / (0.50849 * 0.69966) ≈ 0.08714
//
// doc2: no query terms → score = 0 (absent from results)

TEST(Ranking, ManualTFIDFVerification) {
    TestCorpus tc({
        {0, "cat sat mat"},
        {1, "cat ate rat"},
        {2, "dog bit man"},
    });

    auto results = tc.search("cat sat");

    // doc 2 must not appear
    ASSERT_GE(results.size(), 2u);
    bool has_doc2 = false;
    for (const auto& r : results) {
        if (r.doc_id == 2) has_doc2 = true;
    }
    EXPECT_FALSE(has_doc2) << "doc2 has no query terms and must not appear";

    // doc0 must rank first
    EXPECT_EQ(results[0].doc_id, 0);
    EXPECT_EQ(results[1].doc_id, 1);

    // Verify scores are in (0, 1]
    for (const auto& r : results) {
        EXPECT_GT(r.score, 0.0);
        EXPECT_LE(r.score, 1.0 + 1e-9);
    }

    // Score of doc0 should be substantially higher than doc1
    EXPECT_GT(results[0].score, results[1].score * 2.0)
        << "doc0 (two query matches) should score much higher than doc1 (one match)";
}
