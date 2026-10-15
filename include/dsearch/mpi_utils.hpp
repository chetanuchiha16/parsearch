#pragma once

#include <mpi.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>

#include "dsearch/search_engine.hpp"
#include "dsearch/statistics.hpp"

namespace dsearch {

/// Utility functions for serializing and transmitting dsearch data structures
/// across MPI ranks via native MPI message passing primitives.
class MPISerializer {
public:
    /// Serialize a term->count document frequency map into a flat string payload:
    /// "term1\tcount1\nterm2\tcount2\n"
    static std::string serialize_df_map(const std::unordered_map<std::string, int>& df_map) {
        std::ostringstream oss;
        for (const auto& [term, count] : df_map) {
            oss << term << '\t' << count << '\n';
        }
        return oss.str();
    }

    /// Deserialize a flat string payload back into a term->count document frequency map.
    static std::unordered_map<std::string, int> deserialize_df_map(const std::string& data) {
        std::unordered_map<std::string, int> df_map;
        std::istringstream iss(data);
        std::string line;
        while (std::getline(iss, line)) {
            if (line.empty()) continue;
            auto tab_pos = line.find('\t');
            if (tab_pos != std::string::npos) {
                std::string term = line.substr(0, tab_pos);
                int count = std::stoi(line.substr(tab_pos + 1));
                df_map[term] = count;
            }
        }
        return df_map;
    }

    /// Serialize Statistics (total_docs, DF, IDF) into a flat string payload.
    static std::string serialize_statistics(const Statistics& stats) {
        std::ostringstream oss;
        oss << stats.total_docs() << '\n';
        for (const auto& [term, idf_val] : stats.idf_map()) {
            oss << term << '\t' << stats.df(term) << '\t' << idf_val << '\n';
        }
        return oss.str();
    }

    /// Deserialize a flat string payload back into a Statistics object.
    static Statistics deserialize_statistics(const std::string& data) {
        Statistics stats;
        std::istringstream iss(data);
        std::string line;
        if (std::getline(iss, line) && !line.empty()) {
            stats.total_docs_ = std::stoi(line);
        }
        while (std::getline(iss, line)) {
            if (line.empty()) continue;
            auto p1 = line.find('\t');
            if (p1 == std::string::npos) continue;
            auto p2 = line.find('\t', p1 + 1);
            if (p2 == std::string::npos) continue;

            std::string term = line.substr(0, p1);
            int df = std::stoi(line.substr(p1 + 1, p2 - p1 - 1));
            double idf = std::stod(line.substr(p2 + 1));

            stats.df_[term] = df;
            stats.idf_[term] = idf;
        }
        return stats;
    }

    /// Serialize search results: "doc_id\tscore\tpath\n"
    static std::string serialize_results(const std::vector<SearchResult>& results) {
        std::ostringstream oss;
        for (const auto& r : results) {
            oss << r.doc_id << '\t' << r.score << '\t' << r.path << '\n';
        }
        return oss.str();
    }

    /// Deserialize search results.
    static std::vector<SearchResult> deserialize_results(const std::string& data) {
        std::vector<SearchResult> results;
        std::istringstream iss(data);
        std::string line;
        while (std::getline(iss, line)) {
            if (line.empty()) continue;
            auto p1 = line.find('\t');
            if (p1 == std::string::npos) continue;
            auto p2 = line.find('\t', p1 + 1);
            if (p2 == std::string::npos) continue;

            SearchResult r;
            r.doc_id = std::stoi(line.substr(0, p1));
            r.score = std::stod(line.substr(p1 + 1, p2 - p1 - 1));
            r.path = line.substr(p2 + 1);
            results.push_back(r);
        }
        return results;
    }

    /// Broadcast a std::string from root to all ranks in comm.
    static void bcast_string(std::string& s, int root, MPI_Comm comm) {
        int len = static_cast<int>(s.size());
        MPI_Bcast(&len, 1, MPI_INT, root, comm);
        s.resize(len);
        if (len > 0) {
            MPI_Bcast(s.data(), len, MPI_CHAR, root, comm);
        }
    }

    /// Send a std::string point-to-point.
    static void send_string(const std::string& s, int dest, int tag, MPI_Comm comm) {
        int len = static_cast<int>(s.size());
        MPI_Send(&len, 1, MPI_INT, dest, tag, comm);
        if (len > 0) {
            MPI_Send(s.data(), len, MPI_CHAR, dest, tag, comm);
        }
    }

    /// Receive a std::string point-to-point.
    static std::string recv_string(int source, int tag, MPI_Comm comm) {
        int len = 0;
        MPI_Status status;
        MPI_Recv(&len, 1, MPI_INT, source, tag, comm, &status);
        std::string s(len, '\0');
        if (len > 0) {
            MPI_Recv(s.data(), len, MPI_CHAR, source, tag, comm, &status);
        }
        return s;
    }
};

} // namespace dsearch
