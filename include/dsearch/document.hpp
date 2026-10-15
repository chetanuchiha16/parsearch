#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace dsearch {

/// Represents a single source document loaded from the filesystem.
struct Document {
    int                   id;      ///< Stable integer ID assigned by sorted-path order
    std::filesystem::path path;   ///< Path to the source file
    std::string           content; ///< Raw UTF-8 text content
};

/// Read the full content of a file into a string.
/// Returns an empty string if the file cannot be opened or is empty.
std::string read_file(const std::filesystem::path& path);

/// Recursively scan a directory for .txt and .md files and load them.
///
/// IDs are assigned in ascending lexicographic path order so that the
/// same directory always produces the same ID assignment (deterministic).
///
/// @param dir  Root directory to scan (must exist and be a directory)
/// @returns    Loaded documents sorted by path; empty files are skipped
/// @throws     std::runtime_error if dir does not exist or is not a directory
std::vector<Document> load_documents(const std::filesystem::path& dir);

} // namespace dsearch
