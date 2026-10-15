# dsearch — Distributed Parallel Semantic Document Search Engine

A C++20 systems project that searches a large collection of local text documents
using parallel processing, distributed workers, and optional semantic search.

## Build

### Requirements

| Tool | Minimum version |
|------|-----------------|
| CMake | 3.20 |
| C++ compiler | GCC 11 / Clang 13 (C++20 required) |
| Git | Any recent version (for FetchContent) |

### Configure and build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

CLI11 and GoogleTest are fetched automatically by CMake on first configure.

## Quick start

### Generate a sample corpus

```sh
python3 scripts/generate_corpus.py --output-dir ./data --count 25
```

### Index documents

```sh
./build/dsearch index ./data
```

### Search

```sh
./build/dsearch search "database crash recovery" --corpus ./data --top-k 5
```

### Corpus statistics

```sh
./build/dsearch stats ./data
```

## Run tests

```sh
cmake --build build --target run_tests
ctest --test-dir build --output-on-failure
```

## Project phases

| Phase | Description | Status |
|-------|-------------|--------|
| 01 | Sequential TF-IDF engine | ✅ Complete |
| 02 | Persistent binary index | ✅ Complete |
| 03 | OpenMP parallelization | ✅ Complete |
| 04 | Distributed workers & partitioning | ✅ Complete |
| 05 | Storage & query optimization | ✅ Complete |
| 06 | Semantic search *(optional)* | ⏭️ Skipped |
| 07 | Benchmark + hardening | ✅ Complete |

## Usage & CLI Commands

```sh
# Build index and persist to disk (supports --threads <N>)
./build/dsearch index ./data --index-dir ./index --threads 4

# Inspect persisted binary index metadata
./build/dsearch status --index-dir ./index

# Search directly from persisted index
./build/dsearch search "database crash recovery" --index-dir ./index --top-k 5

# Search with distributed workers
./build/dsearch dist-search "distributed consensus raft" --corpus ./data --workers 4 --top-k 5

# Show corpus statistics
./build/dsearch stats --index-dir ./index

# Run benchmark scaling harness (1, 2, 4, 8, 12 threads)
./build/dsearch_benchmark --corpus ./data --threads 1 2 4 8 12
```

## Architecture (Phase 01)

```
CLI (main.cpp)
  └── load_documents()        document.cpp
  └── Tokenizer::tokenize()   tokenizer.cpp
  └── InvertedIndex           inverted_index.cpp
  └── Statistics::compute()   statistics.cpp
  └── keyword_search()        search_engine.cpp
```

## TF-IDF formulas

```
tf_weight(t, d) = 1 + log10( count(t, d) )    [log-normalised]
idf(t)          = log10( N / df(t) )           [standard IDF]
w(t, d)         = tf_weight(t, d) * idf(t)     [TF-IDF weight]
score(q, d)     = dot(q, d) / ( ||q|| * ||d|| ) [cosine similarity]
```

## Repository structure

```
parsearch/
├── CMakeLists.txt
├── README.md
├── include/dsearch/       # Public headers
├── src/                   # Implementation files + main
├── tests/                 # Unit tests (GoogleTest)
├── benchmarks/            # Performance benchmarks (Phase 07)
├── scripts/               # Utility scripts
├── data/                  # Sample documents
└── index/                 # Generated index files (Phase 02+)
```
