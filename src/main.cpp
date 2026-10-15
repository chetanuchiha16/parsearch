#include <CLI/CLI.hpp>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "dsearch/document.hpp"
#include "dsearch/inverted_index.hpp"
#include "dsearch/search_engine.hpp"
#include "dsearch/statistics.hpp"
#include "dsearch/tokenizer.hpp"

namespace fs = std::filesystem;
using namespace dsearch;

// ── Shared helpers ────────────────────────────────────────────────────────

/// Build an in-memory index + stats from a corpus directory.
/// Returns the number of documents indexed.
static int build_index(const fs::path& corpus_dir,
                        InvertedIndex& index,
                        Statistics&    stats,
                        Tokenizer&     tokenizer) {
    auto docs = load_documents(corpus_dir);
    if (docs.empty()) return 0;

    for (const auto& doc : docs) {
        auto tokens = tokenizer.tokenize(doc.content);
        index.add_document(doc.id, tokens, doc.path.string());
    }

    stats.compute(index);
    index.compute_doc_norms(stats);
    return static_cast<int>(docs.size());
}

// ── Subcommand: index ─────────────────────────────────────────────────────

static int cmd_index(const std::string& corpus) {
    const fs::path corpus_dir{corpus};
    std::cout << "[dsearch] Scanning: " << corpus_dir << "\n";

    Tokenizer     tokenizer;
    InvertedIndex index;
    Statistics    stats;

    const int n = build_index(corpus_dir, index, stats, tokenizer);
    if (n == 0) {
        std::cerr << "[dsearch] No .txt or .md documents found in: " << corpus_dir << "\n";
        return 1;
    }

    // Count unique terms (vocabulary size)
    const int vocab_size = static_cast<int>(index.get_index().size());

    std::cout << "[dsearch] Indexed      : " << n         << " documents\n";
    std::cout << "[dsearch] Vocabulary   : " << vocab_size << " unique terms\n";
    std::cout << "[dsearch] Note         : persistence not yet implemented (Phase 02)\n";
    return 0;
}

// ── Subcommand: search ────────────────────────────────────────────────────

static int cmd_search(const std::string& corpus,
                       const std::string& query,
                       int                top_k) {
    const fs::path corpus_dir{corpus};

    Tokenizer     tokenizer;
    InvertedIndex index;
    Statistics    stats;

    const int n = build_index(corpus_dir, index, stats, tokenizer);
    if (n == 0) {
        std::cerr << "[dsearch] No documents found in: " << corpus_dir << "\n";
        return 1;
    }

    const auto results = keyword_search(query, index, stats, tokenizer, top_k);

    if (results.empty()) {
        std::cout << "No results found for: \"" << query << "\"\n";
        return 0;
    }

    int rank = 1;
    for (const auto& r : results) {
        std::cout << std::setw(2) << rank++ << ". "
                  << r.path
                  << "    score=" << std::fixed << std::setprecision(4) << r.score
                  << "\n";
    }
    return 0;
}

// ── Subcommand: stats ─────────────────────────────────────────────────────

static int cmd_stats(const std::string& corpus) {
    const fs::path corpus_dir{corpus};

    Tokenizer     tokenizer;
    InvertedIndex index;
    Statistics    stats;

    const int n = build_index(corpus_dir, index, stats, tokenizer);
    if (n == 0) {
        std::cerr << "[dsearch] No documents found in: " << corpus_dir << "\n";
        return 1;
    }

    const int vocab_size = static_cast<int>(index.get_index().size());

    // Count total tokens across all docs
    long long total_tokens = 0;
    for (const auto& [id, meta] : index.get_docs()) {
        total_tokens += meta.length;
    }

    std::cout << "Corpus statistics\n";
    std::cout << "  Documents  : " << n           << "\n";
    std::cout << "  Vocabulary : " << vocab_size   << " unique terms\n";
    std::cout << "  Total tokens: " << total_tokens << "\n";
    std::cout << "  Avg tokens/doc: "
              << std::fixed << std::setprecision(1)
              << (n > 0 ? static_cast<double>(total_tokens) / n : 0.0)
              << "\n";
    return 0;
}

// ── main ──────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    CLI::App app{"dsearch — Distributed Parallel Semantic Document Search Engine",
                 "dsearch"};
    app.require_subcommand(1);

    // ── index ──────────────────────────────────────────────────────────────
    auto* sub_index  = app.add_subcommand("index",  "Build a search index from a document directory");
    std::string index_corpus;
    sub_index->add_option("corpus", index_corpus, "Path to document directory")->required();

    // ── search ─────────────────────────────────────────────────────────────
    auto* sub_search = app.add_subcommand("search", "Search the indexed corpus");
    std::string search_corpus, query;
    int top_k = 10;
    sub_search->add_option("query",    query,         "Search query")->required();
    sub_search->add_option("--corpus", search_corpus, "Path to document directory")->required();
    sub_search->add_option("--top-k",  top_k,         "Number of results to return")->default_val(10);

    // ── stats ──────────────────────────────────────────────────────────────
    auto* sub_stats  = app.add_subcommand("stats",  "Show corpus statistics");
    std::string stats_corpus;
    sub_stats->add_option("corpus", stats_corpus, "Path to document directory")->required();

    // ── status ─────────────────────────────────────────────────────────────
    auto* sub_status = app.add_subcommand("status", "Show engine version and status");
    (void)sub_status;

    CLI11_PARSE(app, argc, argv);

    try {
        if (sub_index->parsed())  return cmd_index(index_corpus);
        if (sub_search->parsed()) return cmd_search(search_corpus, query, top_k);
        if (sub_stats->parsed())  return cmd_stats(stats_corpus);
        if (sub_status->parsed()) {
            std::cout << "dsearch v0.1.0 (Phase 01 — Sequential Engine)\n";
            std::cout << "Index persistence: not yet implemented (Phase 02)\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
