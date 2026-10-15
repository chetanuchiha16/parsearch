#include "dsearch/document.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace dsearch {

std::string read_file(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::vector<Document> load_documents(const std::filesystem::path& dir) {
    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
        throw std::runtime_error("Not a valid directory: " + dir.string());
    }

    // Collect all .txt and .md paths
    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        const auto ext = entry.path().extension().string();
        if (ext == ".txt" || ext == ".md") {
            paths.push_back(entry.path());
        }
    }

    // Sort for deterministic ID assignment across runs
    std::sort(paths.begin(), paths.end());

    std::vector<Document> docs;
    docs.reserve(paths.size());

    int next_id = 0;
    for (auto& p : paths) {
        auto content = read_file(p);
        if (content.empty()) continue; // skip unreadable or empty files
        docs.push_back({next_id++, p, std::move(content)});
    }

    return docs;
}

} // namespace dsearch
