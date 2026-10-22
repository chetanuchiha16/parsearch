# dsearch — Distributed Parallel Semantic Document Search Engine

A C++20 systems project that searches a large collection of local text documents
using parallel processing, distributed workers, and optional semantic search.

## Build

### Requirements

| Tool | Minimum version | Notes |
|------|-----------------|-------|
| CMake | 3.20 | Build configuration |
| C++ compiler | GCC 11 / Clang 13 | C++20 required |
| OpenMP | 4.5+ | Intra-node multi-threading |
| OpenMPI / MPICH | 4.0+ | Inter-node process distribution (`openmpi-devel`) |
| Git | Any recent | FetchContent for CLI11 & GoogleTest |

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

## Interactive Web UI Dashboard

A standalone web UI dashboard is provided for live visual demonstration:

```sh
python3 ui/app.py
```
Then open **`http://localhost:8080`** in any web browser.

**Features:**
- Real-time TF-IDF search across the corpus with snippets and relevance scores.
- Dynamic paradigm switcher: **Serial**, **OpenMP**, and **Hybrid MPI**.
- On-the-fly tuning of OpenMP cores and MPI rank partitions.
- Live **3-Way Benchmark Race** with interactive Chart.js latency graphs.
- Document content viewer modal.

## Execution Paradigms

`dsearch` provides three execution paradigms to evaluate parallel and distributed systems trade-offs:

1. **Serial (1 Thread, 1 Process)**:
   ```bash
   ./build/dsearch search "database crash recovery" --corpus ./data --top-k 5
   ```

2. **Parallel (Shared-Memory Multi-Core OpenMP)**:
   ```bash
   OMP_NUM_THREADS=6 ./build/dsearch search "database crash recovery" --corpus ./data --top-k 5
   ```

3. **Hybrid Distributed (Multi-Process MPI + Intra-Node OpenMP)**:
   ```bash
   mpirun -np 4 ./build/dsearch_mpi -c ./data -q "database crash recovery" -k 5 -t 2
   ```

## Three-Way Performance Benchmark

Run the rigorous automated benchmark comparing all three paradigms on the same machine:
```bash
mpirun -np 4 ./build/dsearch_compare ./bench_compare_data
```

## Run Tests
```bash
./build/run_tests
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

## Complete CLI & Executables Reference

The project produces 4 distinct executables:

### 1. `dsearch` (Single-Node Multi-Threaded Engine)

```sh
# 1. Ingest corpus and build binary index (optional --threads <N>, --index-dir <path>)
./build/dsearch index ./data --index-dir ./index --threads 6

# 2. Search on-the-fly directly from source documents
./build/dsearch search "database crash recovery" --corpus ./data --top-k 5 --threads 6

# 3. Search high-speed from a pre-built persisted binary index
./build/dsearch search "database crash recovery" --index-dir ./index --top-k 5

# 4. Multi-worker simulated distributed search (in-process partitioning)
./build/dsearch dist-search "distributed consensus raft" --corpus ./data --workers 4 --top-k 5

# 5. Display corpus word count, vocabulary size, and document statistics
./build/dsearch stats ./data
# Or inspect from saved binary index:
./build/dsearch stats --index-dir ./index

# 6. Verify binary index header integrity and format version
./build/dsearch status --index-dir ./index
```

### 2. `dsearch_mpi` (Hybrid Multi-Process OpenMPI + OpenMP)

```sh
# Run cluster of 4 MPI ranks with 2 OpenMP threads per rank
mpirun -np 4 ./build/dsearch_mpi -c ./data -q "database crash recovery" -k 5 -t 2

# Supported flags:
#   -c, --corpus   Path to document corpus directory (default: ./data)
#   -q, --query    Search query keyword string
#   -k, --top-k    Number of top results to return (default: 10)
#   -t, --threads  OpenMP worker threads per MPI process (default: 2)
```

### 3. `dsearch_compare` (Three-Way Rigorous Benchmark)

```sh
# Runs Serial vs OpenMP vs Hybrid MPI side-by-side with identical repeated trials
mpirun -np 4 ./build/dsearch_compare ./bench_compare_data
```

### 4. `dsearch_benchmark` (OpenMP Thread Scaling Harness)

```sh
# Measures speedup and Amdahl scaling across specified thread counts
./build/dsearch_benchmark --corpus ./data --threads 1 2 4 6 12 --repeats 5
```

### 5. `scripts/generate_corpus.py` (Synthetic Corpus Generator)

```sh
# Generate 1,000 synthetic documents with repeated content to simulate larger files
python3 scripts/generate_corpus.py --output-dir ./bench_compare_data --count 1000 --multiplier 15
```

## Hybrid Distributed Architecture
 
```
                      ┌─────────────────────────────────────────┐
                      │        MPI Coordinator (Rank 0)         │
                      └────────────────────┬────────────────────┘
                                           │
                    MPI Message Passing    │  MPI_Bcast (Query / Global IDF)
                    (Network / Unix Pipes) │  MPI_Send / MPI_Recv (Top-k lists)
                                           │
         ┌──────────────────┬──────────────┴─────┬──────────────────┐
         │                  │                    │                  │
         ▼                  ▼                    ▼                  ▼
  ┌──────────────┐   ┌──────────────┐     ┌──────────────┐   ┌──────────────┐
  │  MPI Rank 0  │   │  MPI Rank 1  │     │  MPI Rank 2  │   │  MPI Rank 3  │
  │ (Partition 0)│   │ (Partition 1)│     │ (Partition 2)│   │ (Partition 3)│
  └──────┬───────┘   └──────┬───────┘     └──────┬───────┘   └──────┬───────┘
         │                  │                    │                  │
   OpenMP Threads     OpenMP Threads       OpenMP Threads     OpenMP Threads
     ┌───┴───┐          ┌───┴───┐            ┌───┴───┐          ┌───┴───┐
    T0      T1         T0      T1           T0      T1         T0      T1
```

## TF-IDF formulas

```
tf_weight(t, d) = 1 + log10( count(t, d) )             [log-normalised]
idf(t)          = log10( (N + 1) / (df(t) + 1) ) + 1   [smoothed IDF]
w(t, d)         = tf_weight(t, d) * idf(t)              [TF-IDF weight]
score(q, d)     = dot(q, d) / ( ||q|| * ||d|| )         [cosine similarity]
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
