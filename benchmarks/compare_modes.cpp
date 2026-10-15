#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "dsearch/coordinator.hpp"
#include "dsearch/document.hpp"
#include "dsearch/inverted_index.hpp"
#include "dsearch/search_engine.hpp"
#include "dsearch/statistics.hpp"
#include "dsearch/tokenizer.hpp"

namespace fs = std::filesystem;
using namespace dsearch;

int main(int argc, char* argv[]) {
    std::string corpus_dir = "./bench_compare_data";
    if (argc > 1) {
        corpus_dir = argv[1];
    }

    std::string query = "database recovery transaction crash log consensus operating systems";
    int top_k = 10;
    int repeats = 5;

    std::cout << "=========================================================================\n";
    std::cout << " dsearch Three-Way Comparison: Serial vs Parallel (OpenMP) vs Distributed\n";
    std::cout << " Corpus : " << corpus_dir << "\n";
    std::cout << " Query  : \"" << query << "\"\n";
    std::cout << " Repeats: " << repeats << "\n";
    std::cout << "=========================================================================\n\n";

    Tokenizer tokenizer;

    // Verify corpus
    auto docs = load_documents(corpus_dir);
    if (docs.empty()) {
        std::cerr << "Error: no documents found in " << corpus_dir << "\n";
        return 1;
    }
    std::cout << "Indexed Corpus Size: " << docs.size() << " documents\n\n";

    // ─────────────────────────────────────────────────────────────────────────
    // 1. SERIAL (1 thread, 1 process)
    // ─────────────────────────────────────────────────────────────────────────
    double serial_index_total = 0.0;
    InvertedIndex serial_index;
    Statistics serial_stats;

    for (int r = 0; r < repeats; ++r) {
        serial_index.clear();
        serial_stats.clear();
        auto t0 = std::chrono::high_resolution_clock::now();
        build_index_parallel(corpus_dir, serial_index, serial_stats, tokenizer, 1);
        auto t1 = std::chrono::high_resolution_clock::now();
        serial_index_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    double serial_index_time = serial_index_total / repeats;

    double serial_query_total = 0.0;
    for (int r = 0; r < repeats * 20; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto res = keyword_search(query, serial_index, serial_stats, tokenizer, top_k, 1);
        auto t1 = std::chrono::high_resolution_clock::now();
        serial_query_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    double serial_query_time = serial_query_total / (repeats * 20);

    // ─────────────────────────────────────────────────────────────────────────
    // 2. PARALLEL (OpenMP across 6 physical CPU cores)
    // ─────────────────────────────────────────────────────────────────────────
    int omp_threads = 6;
    double parallel_index_total = 0.0;
    InvertedIndex parallel_index;
    Statistics parallel_stats;

    for (int r = 0; r < repeats; ++r) {
        parallel_index.clear();
        parallel_stats.clear();
        auto t0 = std::chrono::high_resolution_clock::now();
        build_index_parallel(corpus_dir, parallel_index, parallel_stats, tokenizer, omp_threads);
        auto t1 = std::chrono::high_resolution_clock::now();
        parallel_index_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    double parallel_index_time = parallel_index_total / repeats;

    double parallel_query_total = 0.0;
    for (int r = 0; r < repeats * 20; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto res = keyword_search(query, parallel_index, parallel_stats, tokenizer, top_k, omp_threads);
        auto t1 = std::chrono::high_resolution_clock::now();
        parallel_query_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    double parallel_query_time = parallel_query_total / (repeats * 20);

    // ─────────────────────────────────────────────────────────────────────────
    // 3. DISTRIBUTED (Coordinator + 4 Worker Partitions)
    // ─────────────────────────────────────────────────────────────────────────
    int num_workers = 4;
    DistributedCoordinator coordinator(num_workers);
    auto partitions = coordinator.partition_documents(static_cast<int>(docs.size()));

    double dist_index_total = 0.0;
    std::vector<DistributedWorker> workers;
    Statistics dist_global_stats;

    for (int r = 0; r < repeats; ++r) {
        workers.clear();
        std::vector<std::unordered_map<std::string, int>> worker_dfs;

        auto t0 = std::chrono::high_resolution_clock::now();

        // Pre-bucket documents by partition to avoid re-reading disk N times
        std::vector<std::vector<Document>> worker_doc_buckets(num_workers);
        for (const auto& doc : docs) {
            int target = doc.id % num_workers;
            worker_doc_buckets[target].push_back(doc);
        }

        workers.resize(num_workers, DistributedWorker(0, {}));
        worker_dfs.resize(num_workers);

        // 1. Workers build partition indexes concurrently across worker processes/threads
        #pragma omp parallel for schedule(static) num_threads(num_workers)
        for (int w = 0; w < num_workers; ++w) {
            DistributedWorker worker(partitions[w].worker_rank, partitions[w].doc_ids);
            worker.index_partition(worker_doc_buckets[w], tokenizer);
            worker_dfs[w] = worker.get_local_df();
            workers[w] = std::move(worker);
        }

        // 2. Coordinator aggregates global statistics
        dist_global_stats = coordinator.aggregate_global_statistics(static_cast<int>(docs.size()), worker_dfs);

        // 3. Workers apply global IDF to calibrate local norms concurrently
        #pragma omp parallel for schedule(static) num_threads(num_workers)
        for (int w = 0; w < num_workers; ++w) {
            workers[w].apply_global_statistics(dist_global_stats);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        dist_index_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    double dist_index_time = dist_index_total / repeats;

    double dist_query_total = 0.0;
    for (int r = 0; r < repeats * 20; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // 1. Query dispatched concurrently to all workers
        std::vector<std::vector<SearchResult>> worker_results(num_workers);
        #pragma omp parallel for schedule(static) num_threads(num_workers)
        for (int w = 0; w < num_workers; ++w) {
            worker_results[w] = workers[w].search(query, dist_global_stats, tokenizer, top_k);
        }

        // 2. Coordinator merges top-k lists
        auto final_results = coordinator.merge_worker_results(worker_results, top_k);

        auto t1 = std::chrono::high_resolution_clock::now();
        dist_query_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    double dist_query_time = dist_query_total / (repeats * 20);

    // ─────────────────────────────────────────────────────────────────────────
    // Results Table
    // ─────────────────────────────────────────────────────────────────────────
    std::cout << std::left
              << std::setw(28) << "Execution Mode"
              << std::setw(20) << "Indexing Time (ms)"
              << std::setw(18) << "Index Speedup"
              << std::setw(20) << "Query Latency (ms)"
              << std::setw(18) << "Query Speedup"
              << "\n";
    std::cout << std::string(98, '-') << "\n";

    // Serial
    std::cout << std::left
              << std::setw(28) << "Serial (1 Thread)"
              << std::setw(20) << std::fixed << std::setprecision(2) << serial_index_time
              << std::setw(18) << "1.00x (baseline)"
              << std::setw(20) << std::fixed << std::setprecision(4) << serial_query_time
              << std::setw(18) << "1.00x (baseline)"
              << "\n";

    // Parallel
    double par_idx_speedup = serial_index_time / parallel_index_time;
    double par_qry_speedup = serial_query_time / parallel_query_time;
    std::cout << std::left
              << std::setw(28) << "Parallel (OpenMP 6 Threads)"
              << std::setw(20) << std::fixed << std::setprecision(2) << parallel_index_time
              << std::setw(18) << (std::to_string(par_idx_speedup).substr(0, 4) + "x")
              << std::setw(20) << std::fixed << std::setprecision(4) << parallel_query_time
              << std::setw(18) << (std::to_string(par_qry_speedup).substr(0, 4) + "x")
              << "\n";

    // Distributed
    double dist_idx_speedup = serial_index_time / dist_index_time;
    double dist_qry_speedup = serial_query_time / dist_query_time;
    std::cout << std::left
              << std::setw(28) << "Distributed (4 Workers)"
              << std::setw(20) << std::fixed << std::setprecision(2) << dist_index_time
              << std::setw(18) << (std::to_string(dist_idx_speedup).substr(0, 4) + "x")
              << std::setw(20) << std::fixed << std::setprecision(4) << dist_query_time
              << std::setw(18) << (std::to_string(dist_qry_speedup).substr(0, 4) + "x")
              << "\n";

    std::cout << std::string(98, '-') << "\n";
    return 0;
}
