#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "dsearch/coordinator.hpp"
#include "dsearch/document.hpp"
#include "dsearch/inverted_index.hpp"
#include "dsearch/mpi_utils.hpp"
#include "dsearch/search_engine.hpp"
#include "dsearch/statistics.hpp"
#include "dsearch/tokenizer.hpp"

namespace fs = std::filesystem;
using namespace dsearch;

int main(int argc, char* argv[]) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::string corpus_dir = "./bench_compare_data";
    if (argc > 1) {
        corpus_dir = argv[1];
    }

    std::string query = "database recovery transaction crash log consensus operating systems";
    int top_k = 10;
    int repeats = 5;

    Tokenizer tokenizer;

    // Load docs on root to find size and partitions
    int total_docs = 0;
    std::vector<Document> docs;
    if (rank == 0) {
        docs = load_documents(corpus_dir);
        total_docs = static_cast<int>(docs.size());
        std::cout << "=========================================================================\n";
        std::cout << " dsearch Rigorous Comparison: Serial vs OpenMP vs Hybrid (MPI + OpenMP)\n";
        std::cout << " Corpus       : " << corpus_dir << " (" << total_docs << " documents)\n";
        std::cout << " MPI Ranks    : " << world_size << "\n";
        std::cout << " OpenMP Cores : 6 physical cores\n";
        std::cout << " Query        : \"" << query << "\"\n";
        std::cout << " Repeats      : " << repeats << "\n";
        std::cout << "=========================================================================\n\n";
    }

    MPI_Bcast(&total_docs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (total_docs == 0) {
        MPI_Finalize();
        return 0;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 1. SERIAL (Rank 0 only, 1 thread)
    // ─────────────────────────────────────────────────────────────────────────
    double serial_index_time = 0.0;
    double serial_query_time = 0.0;

    if (rank == 0) {
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
        serial_index_time = serial_index_total / repeats;

        double serial_query_total = 0.0;
        for (int r = 0; r < repeats * 20; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto res = keyword_search(query, serial_index, serial_stats, tokenizer, top_k, 1);
            auto t1 = std::chrono::high_resolution_clock::now();
            serial_query_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        serial_query_time = serial_query_total / (repeats * 20);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 2. SHARED-MEMORY PARALLEL (Rank 0 only, OpenMP 6 Threads)
    // ─────────────────────────────────────────────────────────────────────────
    double parallel_index_time = 0.0;
    double parallel_query_time = 0.0;

    if (rank == 0) {
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
        parallel_index_time = parallel_index_total / repeats;

        double parallel_query_total = 0.0;
        for (int r = 0; r < repeats * 20; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto res = keyword_search(query, parallel_index, parallel_stats, tokenizer, top_k, omp_threads);
            auto t1 = std::chrono::high_resolution_clock::now();
            parallel_query_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        parallel_query_time = parallel_query_total / (repeats * 20);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 3. DISTRIBUTED HYBRID (MPI Ranks + OpenMP Threads per Rank)
    // ─────────────────────────────────────────────────────────────────────────
    DistributedCoordinator coordinator(world_size);
    auto partitions = coordinator.partition_documents(total_docs);

    DistributedWorker worker(rank, partitions[rank].doc_ids);
    worker.index_partition(corpus_dir, tokenizer);
    const auto& local_df = worker.get_local_df();

    double mpi_index_total = 0.0;
    Statistics global_stats;

    for (int r = 0; r < repeats; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::high_resolution_clock::now();

        // 1. Each rank builds its local partition
        DistributedWorker bench_worker(rank, partitions[rank].doc_ids);
        bench_worker.index_partition(corpus_dir, tokenizer);
        const auto& bench_local_df = bench_worker.get_local_df();

        // 2. Exchange local DF maps to Rank 0
        std::string serialized_df = MPISerializer::serialize_df_map(bench_local_df);
        if (rank == 0) {
            std::vector<std::unordered_map<std::string, int>> all_dfs(world_size);
            all_dfs[0] = bench_local_df;
            for (int src = 1; src < world_size; ++src) {
                std::string remote_df_str = MPISerializer::recv_string(src, 100, MPI_COMM_WORLD);
                all_dfs[src] = MPISerializer::deserialize_df_map(remote_df_str);
            }
            global_stats = coordinator.aggregate_global_statistics(total_docs, all_dfs);
        } else {
            MPISerializer::send_string(serialized_df, 0, 100, MPI_COMM_WORLD);
        }

        // Step B: Broadcast global Statistics to all ranks
        std::string global_stats_str;
        if (rank == 0) {
            global_stats_str = MPISerializer::serialize_statistics(global_stats);
        }
        MPISerializer::bcast_string(global_stats_str, 0, MPI_COMM_WORLD);

        if (rank != 0) {
            global_stats = MPISerializer::deserialize_statistics(global_stats_str);
        }

        // Step C: Apply global IDF to recalibrate partition norms
        worker.apply_global_statistics(global_stats);

        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (rank == 0) {
            mpi_index_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
    }
    double mpi_index_time = mpi_index_total / repeats;

    // Distributed Query Benchmarking
    double mpi_query_total = 0.0;
    int q_repeats = repeats * 20;

    for (int r = 0; r < q_repeats; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::high_resolution_clock::now();

        // 1. Local partition search
        auto local_results = worker.search(query, global_stats, tokenizer, top_k);
        std::string serialized_res = MPISerializer::serialize_results(local_results);

        // 2. Gather & merge
        if (rank == 0) {
            std::vector<std::vector<SearchResult>> all_results(world_size);
            all_results[0] = std::move(local_results);

            for (int src = 1; src < world_size; ++src) {
                std::string remote_res_str = MPISerializer::recv_string(src, 200, MPI_COMM_WORLD);
                all_results[src] = MPISerializer::deserialize_results(remote_res_str);
            }

            auto merged_results = coordinator.merge_worker_results(all_results, top_k);
            MPI_Barrier(MPI_COMM_WORLD);
            auto t1 = std::chrono::high_resolution_clock::now();
            mpi_query_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
        } else {
            MPISerializer::send_string(serialized_res, 0, 200, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    double mpi_query_time = mpi_query_total / q_repeats;

    // ─────────────────────────────────────────────────────────────────────────
    // Results Reporting on Rank 0
    // ─────────────────────────────────────────────────────────────────────────
    if (rank == 0) {
        std::cout << std::left
                  << std::setw(32) << "Execution Paradigm"
                  << std::setw(20) << "Indexing Time (ms)"
                  << std::setw(18) << "Index Speedup"
                  << std::setw(20) << "Query Latency (ms)"
                  << std::setw(18) << "Query Speedup"
                  << "\n";
        std::cout << std::string(108, '-') << "\n";

        // Serial
        std::cout << std::left
                  << std::setw(32) << "Serial (1 Process, 1 Thread)"
                  << std::setw(20) << std::fixed << std::setprecision(2) << serial_index_time
                  << std::setw(18) << "1.00x (baseline)"
                  << std::setw(20) << std::fixed << std::setprecision(4) << serial_query_time
                  << std::setw(18) << "1.00x (baseline)"
                  << "\n";

        // Shared Memory Parallel
        double par_idx_speedup = serial_index_time / parallel_index_time;
        double par_qry_speedup = serial_query_time / parallel_query_time;
        std::cout << std::left
                  << std::setw(32) << "Parallel (OpenMP 6 Threads)"
                  << std::setw(20) << std::fixed << std::setprecision(2) << parallel_index_time
                  << std::setw(18) << (std::to_string(par_idx_speedup).substr(0, 4) + "x")
                  << std::setw(20) << std::fixed << std::setprecision(4) << parallel_query_time
                  << std::setw(18) << (std::to_string(par_qry_speedup).substr(0, 4) + "x")
                  << "\n";

        // Hybrid MPI + OpenMP
        double mpi_idx_speedup = serial_index_time / mpi_index_time;
        double mpi_qry_speedup = serial_query_time / mpi_query_time;
        std::string label = "Hybrid (MPI " + std::to_string(world_size) + " Ranks + OpenMP)";
        std::cout << std::left
                  << std::setw(32) << label
                  << std::setw(20) << std::fixed << std::setprecision(2) << mpi_index_time
                  << std::setw(18) << (std::to_string(mpi_idx_speedup).substr(0, 4) + "x")
                  << std::setw(20) << std::fixed << std::setprecision(4) << mpi_query_time
                  << std::setw(18) << (std::to_string(mpi_qry_speedup).substr(0, 4) + "x")
                  << "\n";

        std::cout << std::string(108, '-') << "\n\n";

        std::cout << "Architectural Takeaway:\n";
        std::cout << "1. Shared-Memory OpenMP delivers the lowest query latency (" << std::fixed << std::setprecision(4)
                  << parallel_query_time << " ms) due to zero-copy memory access across all cores.\n";
        std::cout << "2. Hybrid MPI introduces message serialization/IPC latency (" << std::fixed << std::setprecision(4)
                  << mpi_query_time << " ms), but divides total memory consumption by " << world_size
                  << ", enabling scaling across multiple independent physical machines.\n";
    }

    MPI_Finalize();
    return 0;
}
