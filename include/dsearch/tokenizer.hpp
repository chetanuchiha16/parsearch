#pragma once

#include <string>
#include <vector>

namespace dsearch {

/// Converts raw document text into a sequence of normalized terms.
///
/// Rules applied in order:
///   1. Lowercase all alphabetic characters.
///   2. Split on any non-alphanumeric character (treated as a delimiter).
///   3. Discard tokens shorter than 2 characters.
///
/// This tokenizer is intentionally simple and self-contained so that the
/// same normalization rules can be applied at both index time and query time.
class Tokenizer {
public:
    /// Tokenize a block of text and return the normalized term sequence.
    /// The same input always produces the same output (deterministic).
    [[nodiscard]] std::vector<std::string> tokenize(const std::string& text) const;
};

} // namespace dsearch
