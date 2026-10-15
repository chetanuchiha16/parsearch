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
| 01 | Sequential TF-IDF engine | 🔄 In progress |
| 02 | Persistent binary index | ⬜ Planned |
| 03 | OpenMP parallelization | ⬜ Planned |
| 04 | MPI distributed workers | ⬜ Planned |
| 05 | Storage optimization | ⬜ Planned |
| 06 | Semantic search | ⬜ Planned |
| 07 | Benchmarks + hardening | ⬜ Planned |

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
