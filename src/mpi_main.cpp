#include <mpi.h>
#include <omp.h>
#include <CLI/CLI.hpp>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>

#include "dsearch/coordinator.hpp"
#include "dsearch/document.hpp"
#include "dsearch/mpi_utils.hpp"
#include "dsearch/search_engine.hpp"
#include "dsearch/statistics.hpp"
#include "dsearch/tokenizer.hpp"

namespace fs = std::filesystem;
using namespace dsearch;

int main(int argc, char* argv[]) {
    // Initialize MPI with thread support for Hybrid MPI+OpenMP
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::string corpus_dir = "./data";
    std::string query_text;
    int top_k = 10;
    int omp_threads = 2; // worker threads per rank

    if (rank == 0) {
        CLI::App app{"dsearch_mpi — Hybrid Distributed MPI+OpenMP Document Search Engine"};
        app.add_option("-c,--corpus", corpus_dir, "Path to document corpus directory")->default_val("./data");
        app.add_option("-q,--query", query_text, "Search query keyword string");
        app.add_option("-k,--top-k", top_k, "Number of top results to return")->default_val(10);
        app.add_option("-t,--threads", omp_threads, "OpenMP threads per MPI rank")->default_val(2);

        try {
            app.parse(argc, argv);
        } catch (const CLI::ParseError& e) {
            int ret = app.exit(e);
            MPI_Abort(MPI_COMM_WORLD, ret);
            return ret;
        }
    }

    // Broadcast CLI configuration to all worker ranks
    MPISerializer::bcast_string(corpus_dir, 0, MPI_COMM_WORLD);
    MPISerializer::bcast_string(query_text, 0, MPI_COMM_WORLD);
    MPI_Bcast(&top_k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&omp_threads, 1, MPI_INT, 0, MPI_COMM_WORLD);

    omp_set_num_threads(omp_threads);
    Tokenizer tokenizer;

    // Scan corpus files
    std::vector<Document> all_docs;
    int total_docs = 0;
    if (rank == 0) {
        try {
            all_docs = load_documents(corpus_dir);
            total_docs = static_cast<int>(all_docs.size());
            std::cout << "=================================================================\n";
            std::cout << " dsearch Distributed MPI Cluster (" << world_size << " ranks, "
                      << omp_threads << " OpenMP threads/rank)\n";
            std::cout << " Corpus: " << corpus_dir << " (" << total_docs << " documents)\n";
            std::cout << "=================================================================\n";
        } catch (const std::exception& e) {
            std::cerr << "Error loading corpus: " << e.what() << "\n";
            total_docs = 0;
        }
    }

    MPI_Bcast(&total_docs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (total_docs == 0) {
        MPI_Finalize();
        return 0;
    }

    // Partition assignment
    DistributedCoordinator coordinator(world_size);
    auto partitions = coordinator.partition_documents(total_docs);

    // Each rank loads or processes its assigned documents
    auto t_idx_start = std::chrono::high_resolution_clock::now();

    DistributedWorker worker(rank, partitions[rank].doc_ids);
    worker.index_partition(corpus_dir, tokenizer);
    const auto& local_df = worker.get_local_df();

    // ── Phase 1: Aggregate global DF via MPI point-to-point / gather ──
    std::string serialized_df = MPISerializer::serialize_df_map(local_df);

    Statistics global_stats;
    if (rank == 0) {
        std::vector<std::unordered_map<std::string, int>> all_dfs(world_size);
        all_dfs[0] = local_df;

        for (int src = 1; src < world_size; ++src) {
            std::string remote_df_str = MPISerializer::recv_string(src, 100, MPI_COMM_WORLD);
            all_dfs[src] = MPISerializer::deserialize_df_map(remote_df_str);
        }

        global_stats = coordinator.aggregate_global_statistics(total_docs, all_dfs);
    } else {
        MPISerializer::send_string(serialized_df, 0, 100, MPI_COMM_WORLD);
    }

    // ── Phase 2: Broadcast global Statistics to all ranks ──
    std::string global_stats_str;
    if (rank == 0) {
        global_stats_str = MPISerializer::serialize_statistics(global_stats);
    }
    MPISerializer::bcast_string(global_stats_str, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        global_stats = MPISerializer::deserialize_statistics(global_stats_str);
    }

    // Apply global statistics to recalibrate local norms
    worker.apply_global_statistics(global_stats);

    auto t_idx_end = std::chrono::high_resolution_clock::now();
    double idx_ms = std::chrono::duration<double, std::milli>(t_idx_end - t_idx_start).count();

    if (rank == 0) {
        std::cout << "[MPI] Distributed Indexing Completed in " << std::fixed << std::setprecision(2)
                  << idx_ms << " ms\n";
    }

    // ── Query Execution if query provided ──
    if (!query_text.empty()) {
        auto t_query_start = std::chrono::high_resolution_clock::now();

        // Each worker computes local top-k
        auto local_results = worker.search(query_text, global_stats, tokenizer, top_k);
        std::string serialized_res = MPISerializer::serialize_results(local_results);

        if (rank == 0) {
            std::vector<std::vector<SearchResult>> all_results(world_size);
            all_results[0] = std::move(local_results);

            for (int src = 1; src < world_size; ++src) {
                std::string remote_res_str = MPISerializer::recv_string(src, 200, MPI_COMM_WORLD);
                all_results[src] = MPISerializer::deserialize_results(remote_res_str);
            }

            auto merged_results = coordinator.merge_worker_results(all_results, top_k);
            auto t_query_end = std::chrono::high_resolution_clock::now();
            double query_ms = std::chrono::duration<double, std::milli>(t_query_end - t_query_start).count();

            std::cout << "\n[MPI] Query: \"" << query_text << "\" executed in "
                      << std::fixed << std::setprecision(4) << query_ms << " ms\n";
            std::cout << "Top " << merged_results.size() << " Results:\n";
            int rank_idx = 1;
            for (const auto& r : merged_results) {
                std::cout << "  " << std::setw(2) << rank_idx++ << ". ["
                          << std::fixed << std::setprecision(4) << r.score << "] "
                          << r.path << " (doc_id=" << r.doc_id << ")\n";
            }
        } else {
            MPISerializer::send_string(serialized_res, 0, 200, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
