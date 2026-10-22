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

    // Collect supported document and text file paths
    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        const auto ext = entry.path().extension().string();
        // Index standard text/code documents as well as common document formats
        if (ext == ".txt" || ext == ".md" || ext == ".tex" ||
            ext == ".pdf" || ext == ".xls" || ext == ".xlsx" ||
            ext == ".doc" || ext == ".docx" || ext == ".csv" ||
            ext == ".cpp" || ext == ".hpp" || ext == ".cc" ||
            ext == ".h" || ext == ".py" || ext == ".json" ||
            ext == ".html" || ext == ".xml" || ext == ".log") {
            paths.push_back(entry.path());
        }
    }

    // Sort for deterministic ID assignment across runs
    std::sort(paths.begin(), paths.end());

    std::vector<Document> docs;
    docs.reserve(paths.size());

    int next_id = 0;
    for (auto& p : paths) {
        const auto ext = p.extension().string();
        std::string content;

        // For plain text / code files, read the file body
        if (ext == ".txt" || ext == ".md" || ext == ".tex" ||
            ext == ".cpp" || ext == ".hpp" || ext == ".cc" ||
            ext == ".h" || ext == ".py" || ext == ".json" ||
            ext == ".html" || ext == ".xml" || ext == ".csv" || ext == ".log") {
            content = read_file(p);
        }

        // Always prepend the filename so file names and extensions are fully indexed and searchable
        std::string filename_str = p.filename().string();
        if (!content.empty()) {
            content = filename_str + "\n" + content;
        } else {
            content = filename_str;
        }

        if (content.empty()) continue; // skip unreadable or empty files
        docs.push_back({next_id++, p, std::move(content)});
    }

    return docs;
}

} // namespace dsearch
