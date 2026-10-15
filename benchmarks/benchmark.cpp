#include <CLI/CLI.hpp>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>

#include "dsearch/inverted_index.hpp"
#include "dsearch/search_engine.hpp"
#include "dsearch/statistics.hpp"
#include "dsearch/tokenizer.hpp"

namespace fs = std::filesystem;
using namespace dsearch;

int main(int argc, char* argv[]) {
    CLI::App app{"dsearch benchmark harness — measure scaling and speedup across thread counts",
                 "dsearch_benchmark"};

    std::string corpus_dir;
    std::string query = "database recovery transaction crash log consensus";
    std::vector<int> threads_to_test = {1, 2, 4, 8};
    int repetitions = 3;

    app.add_option("--corpus", corpus_dir, "Path to document corpus directory")->required();
    app.add_option("--query", query, "Query to benchmark");
    app.add_option("--threads", threads_to_test, "Thread counts to test (e.g. 1 2 4 8)");
    app.add_option("--repeats", repetitions, "Repetitions per test")->default_val(3);

    CLI11_PARSE(app, argc, argv);

    Tokenizer tokenizer;
    std::cout << "=================================================================\n";
    std::cout << " dsearch Benchmark: OpenMP Parallel Indexing and Scoring\n";
    std::cout << " Corpus : " << corpus_dir << "\n";
    std::cout << " Query  : \"" << query << "\"\n";
    std::cout << " Repeats: " << repetitions << "\n";
    std::cout << "=================================================================\n\n";

    double t1_index_ms = 0.0;
    double t1_query_ms = 0.0;

    std::cout << std::left
              << std::setw(10) << "Threads"
              << std::setw(18) << "Index Time (ms)"
              << std::setw(16) << "Index Speedup"
              << std::setw(16) << "Index Effic."
              << std::setw(18) << "Query Time (ms)"
              << std::setw(16) << "Query Speedup"
              << "\n";
    std::cout << std::string(94, '-') << "\n";

    for (int t : threads_to_test) {
        // Measure Indexing time
        double total_index_time = 0.0;
        InvertedIndex index;
        Statistics stats;

        for (int r = 0; r < repetitions; ++r) {
            index.clear();
            stats.clear();
            auto start = std::chrono::high_resolution_clock::now();
            build_index_parallel(corpus_dir, index, stats, tokenizer, t);
            auto end = std::chrono::high_resolution_clock::now();
            total_index_time += std::chrono::duration<double, std::milli>(end - start).count();
        }
        double avg_index_ms = total_index_time / repetitions;

        // Measure Querying time
        double total_query_time = 0.0;
        for (int r = 0; r < repetitions * 20; ++r) {
            auto start = std::chrono::high_resolution_clock::now();
            auto res = keyword_search(query, index, stats, tokenizer, 10, t);
            auto end = std::chrono::high_resolution_clock::now();
            total_query_time += std::chrono::duration<double, std::milli>(end - start).count();
        }
        double avg_query_ms = total_query_time / (repetitions * 20);

        if (t == threads_to_test.front()) {
            t1_index_ms = avg_index_ms;
            t1_query_ms = avg_query_ms;
        }

        double speedup = t1_index_ms > 0 ? (t1_index_ms / avg_index_ms) : 1.0;
        double efficiency = speedup / t;
        double q_speedup = t1_query_ms > 0 ? (t1_query_ms / avg_query_ms) : 1.0;

        std::cout << std::left
                  << std::setw(10) << t
                  << std::setw(18) << std::fixed << std::setprecision(2) << avg_index_ms
                  << std::setw(16) << std::fixed << std::setprecision(2) << speedup
                  << std::setw(16) << std::fixed << std::setprecision(2) << efficiency
                  << std::setw(18) << std::fixed << std::setprecision(4) << avg_query_ms
                  << std::setw(16) << std::fixed << std::setprecision(2) << q_speedup
                  << "\n";
    }

    std::cout << "\nBenchmark complete.\n";
    return 0;
}
