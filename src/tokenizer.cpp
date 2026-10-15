#include "dsearch/tokenizer.hpp"

#include <cctype>

namespace dsearch {

std::vector<std::string> Tokenizer::tokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    std::string current;

    for (unsigned char c : text) {
        if (std::isalnum(c)) {
            // Accumulate alphanumeric characters, lowercased
            current += static_cast<char>(std::tolower(c));
        } else {
            // Non-alphanumeric acts as a delimiter: flush current token
            if (current.size() >= 2) {
                tokens.push_back(std::move(current));
            }
            current.clear();
        }
    }

    // Flush the final token (no trailing delimiter)
    if (current.size() >= 2) {
        tokens.push_back(std::move(current));
    }

    return tokens;
}

} // namespace dsearch
