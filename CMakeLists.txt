cmake_minimum_required(VERSION 3.20)
project(dsearch VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler warnings
add_compile_options(-Wall -Wextra -Wpedantic)

# ── Dependencies via FetchContent ─────────────────────────────────────────
include(FetchContent)

# CLI11: lightweight header-only argument parsing
FetchContent_Declare(
    CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.4.1
)
FetchContent_MakeAvailable(CLI11)

# GoogleTest: unit testing
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
# Prevent GTest from installing itself system-wide
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

find_package(OpenMP REQUIRED)

# ── dsearch library (all components except main) ──────────────────────────
add_library(dsearch_lib
    src/document.cpp
    src/tokenizer.cpp
    src/inverted_index.cpp
    src/statistics.cpp
    src/search_engine.cpp
    src/index_storage.cpp
    src/coordinator.cpp
)
target_include_directories(dsearch_lib PUBLIC include)
target_compile_features(dsearch_lib PUBLIC cxx_std_20)
target_link_libraries(dsearch_lib PUBLIC OpenMP::OpenMP_CXX)

# ── dsearch executable ────────────────────────────────────────────────────
add_executable(dsearch src/main.cpp)
target_link_libraries(dsearch PRIVATE dsearch_lib CLI11::CLI11)

# ── Tests ─────────────────────────────────────────────────────────────────
enable_testing()
add_executable(run_tests
    tests/tokenizer_test.cpp
    tests/ranking_test.cpp
    tests/storage_test.cpp
    tests/parallel_test.cpp
    tests/distributed_test.cpp
    tests/hardening_test.cpp
)
target_link_libraries(run_tests PRIVATE dsearch_lib GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(run_tests)

# ── Benchmark executables ──────────────────────────────────────────────────
add_executable(dsearch_benchmark benchmarks/benchmark.cpp)
target_link_libraries(dsearch_benchmark PRIVATE dsearch_lib CLI11::CLI11)

add_executable(dsearch_compare benchmarks/compare_modes.cpp)
target_link_libraries(dsearch_compare PRIVATE dsearch_lib)
