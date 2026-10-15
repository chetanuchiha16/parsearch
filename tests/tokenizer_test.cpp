#include <gtest/gtest.h>
#include "dsearch/tokenizer.hpp"

using namespace dsearch;

TEST(Tokenizer, Lowercases) {
    Tokenizer t;
    auto tokens = t.tokenize("Hello World");
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0], "hello");
    EXPECT_EQ(tokens[1], "world");
}

TEST(Tokenizer, SplitsOnPunctuation) {
    Tokenizer t;
    // Each punctuation char is a delimiter
    auto tokens = t.tokenize("hello,world.foo!bar");
    ASSERT_EQ(tokens.size(), 4u);
    EXPECT_EQ(tokens[0], "hello");
    EXPECT_EQ(tokens[1], "world");
    EXPECT_EQ(tokens[2], "foo");
    EXPECT_EQ(tokens[3], "bar");
}

TEST(Tokenizer, FiltersShortTokens) {
    Tokenizer t;
    // "I" (len 1) and "a" (len 1) must be discarded
    auto tokens = t.tokenize("I am a cat");
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0], "am");
    EXPECT_EQ(tokens[1], "cat");
}

TEST(Tokenizer, EmptyInputReturnsEmpty) {
    Tokenizer t;
    EXPECT_TRUE(t.tokenize("").empty());
}

TEST(Tokenizer, WhitespaceOnlyReturnsEmpty) {
    Tokenizer t;
    EXPECT_TRUE(t.tokenize("   \t\n  ").empty());
}

TEST(Tokenizer, RepeatedTermsPreserved) {
    Tokenizer t;
    auto tokens = t.tokenize("cat cat cat");
    ASSERT_EQ(tokens.size(), 3u);
    for (const auto& tok : tokens) {
        EXPECT_EQ(tok, "cat");
    }
}

TEST(Tokenizer, AllPunctuation) {
    Tokenizer t;
    EXPECT_TRUE(t.tokenize("!@#$%^&*()").empty());
}

TEST(Tokenizer, MixedCaseNormalized) {
    Tokenizer t;
    auto tokens = t.tokenize("CamelCase UPPERCASE lowercase");
    // CamelCase → "camelcase", UPPERCASE → "uppercase", lowercase → "lowercase"
    ASSERT_EQ(tokens.size(), 3u);
    EXPECT_EQ(tokens[0], "camelcase");
    EXPECT_EQ(tokens[1], "uppercase");
    EXPECT_EQ(tokens[2], "lowercase");
}

TEST(Tokenizer, NumbersPreserved) {
    Tokenizer t;
    auto tokens = t.tokenize("version 20 release");
    ASSERT_EQ(tokens.size(), 3u);
    EXPECT_EQ(tokens[0], "version");
    EXPECT_EQ(tokens[1], "20");
    EXPECT_EQ(tokens[2], "release");
}

TEST(Tokenizer, LeadingTrailingDelimiters) {
    Tokenizer t;
    auto tokens = t.tokenize("  ,hello,world,  ");
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0], "hello");
    EXPECT_EQ(tokens[1], "world");
}
