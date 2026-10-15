#include <CLI/CLI.hpp>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "dsearch/coordinator.hpp"
#include "dsearch/document.hpp"
#include "dsearch/index_storage.hpp"
#include "dsearch/inverted_index.hpp"
#include "dsearch/search_engine.hpp"
#include "dsearch/statistics.hpp"
#include "dsearch/tokenizer.hpp"

namespace fs = std::filesystem;
using namespace dsearch;

// ── Shared helpers ────────────────────────────────────────────────────────

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

static int cmd_index(const std::string& corpus, const std::string& index_dir, int threads) {
    const fs::path corpus_dir{corpus};
    std::cout << "[dsearch] Scanning: " << corpus_dir << "\n";

    Tokenizer     tokenizer;
    InvertedIndex index;
    Statistics    stats;

    const int n = build_index_parallel(corpus_dir, index, stats, tokenizer, threads);
    if (n == 0) {
        std::cerr << "[dsearch] No .txt or .md documents found in: " << corpus_dir << "\n";
        return 1;
    }

    const int vocab_size = static_cast<int>(index.get_index().size());
    std::cout << "[dsearch] Indexed      : " << n         << " documents\n";
    std::cout << "[dsearch] Vocabulary   : " << vocab_size << " unique terms\n";

    if (!index_dir.empty()) {
        fs::path out_dir{index_dir};
        IndexStorage::save(out_dir, index, stats);
        std::cout << "[dsearch] Persisted to : " << out_dir << "/index.bin\n";
    }

    return 0;
}

// ── Subcommand: search ────────────────────────────────────────────────────

static int cmd_search(const std::string& corpus,
                      const std::string& index_dir,
                      const std::string& query,
                      int                top_k,
                      int                threads) {
    Tokenizer     tokenizer;
    InvertedIndex index;
    Statistics    stats;

    if (!index_dir.empty()) {
        fs::path p_index{index_dir};
        IndexStorage::load(p_index, index, stats);
    } else if (!corpus.empty()) {
        fs::path corpus_dir{corpus};
        const int n = build_index_parallel(corpus_dir, index, stats, tokenizer, threads);
        if (n == 0) {
            std::cerr << "[dsearch] No documents found in: " << corpus_dir << "\n";
            return 1;
        }
    } else {
        std::cerr << "[dsearch] Error: either --index-dir or --corpus must be provided\n";
        return 1;
    }

    const auto results = keyword_search(query, index, stats, tokenizer, top_k, threads);

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

static int cmd_stats(const std::string& corpus, const std::string& index_dir) {
    Tokenizer     tokenizer;
    InvertedIndex index;
    Statistics    stats;

    if (!index_dir.empty()) {
        IndexStorage::load(fs::path{index_dir}, index, stats);
    } else if (!corpus.empty()) {
        const int n = build_index_parallel(fs::path{corpus}, index, stats, tokenizer);
        if (n == 0) {
            std::cerr << "[dsearch] No documents found in: " << corpus << "\n";
            return 1;
        }
    } else {
        std::cerr << "[dsearch] Error: either --index-dir or corpus must be provided\n";
        return 1;
    }

    const int n = index.total_docs();
    const int vocab_size = static_cast<int>(index.get_index().size());

    long long total_tokens = 0;
    for (const auto& [id, meta] : index.get_docs()) {
        total_tokens += meta.length;
    }

    std::cout << "Corpus statistics\n";
    std::cout << "  Documents   : " << n           << "\n";
    std::cout << "  Vocabulary  : " << vocab_size   << " unique terms\n";
    std::cout << "  Total tokens: " << total_tokens << "\n";
    std::cout << "  Avg tokens/doc: "
              << std::fixed << std::setprecision(1)
              << (n > 0 ? static_cast<double>(total_tokens) / n : 0.0)
              << "\n";
    return 0;
}

// ── Subcommand: status ────────────────────────────────────────────────────

static int cmd_status(const std::string& index_dir) {
    std::cout << "dsearch v0.3.0 (Phase 03 — OpenMP Parallel Engine)\n";
    if (!index_dir.empty()) {
        try {
            auto header = IndexStorage::read_header(fs::path{index_dir});
            std::cout << "Index status for: " << index_dir << "\n";
            std::cout << "  Format version: " << header.version << "\n";
            std::cout << "  Indexed docs  : " << header.num_docs << "\n";
            std::cout << "  Vocabulary    : " << header.vocab_size << " terms\n";
        } catch (const std::exception& e) {
            std::cerr << "  Index check failed: " << e.what() << "\n";
            return 1;
        }
    }
    return 0;
}

// ── Subcommand: dist-search ───────────────────────────────────────────────

static int cmd_dist_search(const std::string& corpus,
                           const std::string& query,
                           int                top_k,
                           int                num_workers) {
    fs::path corpus_dir{corpus};
    Tokenizer tokenizer;
    auto docs = load_documents(corpus_dir);
    if (docs.empty()) {
        std::cerr << "[dsearch] No documents found in: " << corpus_dir << "\n";
        return 1;
    }

    DistributedCoordinator coordinator(num_workers);
    auto partitions = coordinator.partition_documents(static_cast<int>(docs.size()));

    std::cout << "[dsearch-dist] Distributing " << docs.size() << " documents across "
              << num_workers << " workers\n";

    // Initialize and run worker indexing
    std::vector<DistributedWorker> workers;
    std::vector<std::unordered_map<std::string, int>> worker_dfs;

    for (const auto& part : partitions) {
        DistributedWorker w(part.worker_rank, part.doc_ids);
        w.index_partition(corpus_dir, tokenizer);
        worker_dfs.push_back(w.get_local_df());
        workers.push_back(std::move(w));
    }

    // Coordinator aggregates global statistics
    Statistics global_stats = coordinator.aggregate_global_statistics(static_cast<int>(docs.size()), worker_dfs);

    // Apply global statistics to workers
    for (auto& w : workers) {
        w.apply_global_statistics(global_stats);
    }

    // Dispatch query to each worker and gather local top-k
    std::vector<std::vector<SearchResult>> worker_results;
    for (const auto& w : workers) {
        worker_results.push_back(w.search(query, global_stats, tokenizer, top_k));
    }

    // Coordinator merges and re-ranks top-k
    auto final_results = coordinator.merge_worker_results(worker_results, top_k);

    if (final_results.empty()) {
        std::cout << "No results found for: \"" << query << "\"\n";
        return 0;
    }

    int rank = 1;
    for (const auto& r : final_results) {
        std::cout << std::setw(2) << rank++ << ". "
                  << r.path
                  << "    score=" << std::fixed << std::setprecision(4) << r.score
                  << "\n";
    }
    return 0;
}

// ── main ──────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    CLI::App app{"dsearch — Distributed Parallel Semantic Document Search Engine",
                 "dsearch"};
    app.require_subcommand(1);

    // ── index ──────────────────────────────────────────────────────────────
    auto* sub_index  = app.add_subcommand("index",  "Build a search index from a document directory");
    std::string index_corpus, index_dir_opt;
    int index_threads = 0;
    sub_index->add_option("corpus", index_corpus, "Path to document directory")->required();
    sub_index->add_option("--index-dir", index_dir_opt, "Directory to persist binary index");
    sub_index->add_option("--threads",   index_threads, "Number of OpenMP worker threads (0 = auto)");

    // ── search ─────────────────────────────────────────────────────────────
    auto* sub_search = app.add_subcommand("search", "Search the indexed corpus");
    std::string search_corpus, search_index_dir, query;
    int top_k = 10;
    int search_threads = 0;
    sub_search->add_option("query",        query,            "Search query")->required();
    sub_search->add_option("--corpus",     search_corpus,    "Path to document directory");
    sub_search->add_option("--index-dir",  search_index_dir, "Path to directory containing persisted index");
    sub_search->add_option("--top-k",      top_k,            "Number of results to return")->default_val(10);
    sub_search->add_option("--threads",    search_threads,   "Number of OpenMP worker threads (0 = auto)");

    // ── dist-search ────────────────────────────────────────────────────────
    auto* sub_dist = app.add_subcommand("dist-search", "Search across distributed worker partitions");
    std::string dist_corpus, dist_query;
    int dist_top_k = 10;
    int num_workers = 4;
    sub_dist->add_option("query",       dist_query,   "Search query")->required();
    sub_dist->add_option("--corpus",    dist_corpus,  "Path to document directory")->required();
    sub_dist->add_option("--workers",   num_workers,  "Number of distributed worker partitions")->default_val(4);
    sub_dist->add_option("--top-k",     dist_top_k,   "Number of results to return")->default_val(10);

    // ── stats ──────────────────────────────────────────────────────────────
    auto* sub_stats  = app.add_subcommand("stats",  "Show corpus statistics");
    std::string stats_corpus, stats_index_dir;
    sub_stats->add_option("corpus",        stats_corpus,     "Path to document directory");
    sub_stats->add_option("--index-dir",   stats_index_dir,  "Path to persisted index directory");

    // ── status ─────────────────────────────────────────────────────────────
    auto* sub_status = app.add_subcommand("status", "Show engine and index status");
    std::string status_index_dir;
    sub_status->add_option("--index-dir",  status_index_dir, "Path to index directory to inspect");

    CLI11_PARSE(app, argc, argv);

    try {
        if (sub_index->parsed())  return cmd_index(index_corpus, index_dir_opt, index_threads);
        if (sub_search->parsed()) return cmd_search(search_corpus, search_index_dir, query, top_k, search_threads);
        if (sub_dist->parsed())   return cmd_dist_search(dist_corpus, dist_query, dist_top_k, num_workers);
        if (sub_stats->parsed())  return cmd_stats(stats_corpus, stats_index_dir);
        if (sub_status->parsed()) return cmd_status(status_index_dir);
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
