#!/usr/bin/env python3
"""Generate a synthetic text corpus for testing dsearch.

Usage:
    python scripts/generate_corpus.py --output-dir ./data --count 25
"""
from __future__ import annotations

import argparse
import os
import random

# Each topic has a name and a pool of sentences
TOPICS: list[tuple[str, list[str]]] = [
    ("database", [
        "Databases store structured data in tables and rows.",
        "SQL is the standard language for relational database queries.",
        "Indexes speed up query execution by reducing full table scans.",
        "Transactions ensure atomicity, consistency, isolation, and durability.",
        "Write-ahead logging is used for crash recovery in databases.",
        "A database crash can be recovered using transaction logs.",
        "Checkpoints allow the database to limit redo log replay on restart.",
        "Row-level locking improves concurrency in write-heavy workloads.",
        "B-tree indexes support efficient range queries on sorted data.",
        "Database replication copies data to multiple nodes for high availability.",
        "A storage engine manages how data is physically written to disk.",
        "The buffer pool caches frequently accessed pages in memory.",
    ]),
    ("operating-systems", [
        "An operating system manages CPU, memory, and IO resources.",
        "Processes are isolated address spaces managed by the kernel.",
        "Virtual memory allows programs to use more memory than physically installed.",
        "File systems organize data into directories and files on disk.",
        "The scheduler decides which process runs on which CPU core.",
        "System calls provide the interface between user programs and the kernel.",
        "Semaphores and mutexes are used to synchronize concurrent threads.",
        "Demand paging loads memory pages only when they are accessed.",
        "Interrupts notify the CPU of hardware events like IO completion.",
        "Page faults occur when a program accesses a page not in physical memory.",
        "The kernel maintains a process table to track running programs.",
        "Context switching saves and restores CPU registers between processes.",
    ]),
    ("networking", [
        "TCP provides reliable ordered delivery of data between hosts.",
        "UDP is a connectionless protocol suitable for low-latency applications.",
        "The IP protocol routes packets across interconnected networks.",
        "DNS translates human-readable domain names to IP addresses.",
        "HTTP is the application layer protocol for web communication.",
        "TLS encrypts network traffic to protect data in transit.",
        "A router forwards packets between different network segments.",
        "Bandwidth measures the maximum data transfer rate of a link.",
        "Latency is the time for a packet to travel from source to destination.",
        "Load balancers distribute traffic across multiple servers.",
        "The three-way handshake establishes a TCP connection.",
        "Congestion control prevents network collapse under heavy load.",
    ]),
    ("parallel-computing", [
        "OpenMP provides a shared-memory parallel programming model for C and C++.",
        "Thread-level parallelism splits work across multiple CPU cores.",
        "Race conditions occur when multiple threads access shared data without synchronization.",
        "A mutex prevents concurrent access to a critical section of code.",
        "SIMD instructions process multiple data elements in a single CPU operation.",
        "Cache coherence ensures all CPU cores see a consistent memory state.",
        "Amdahl's law limits the maximum speedup from parallelization.",
        "MPI is the standard interface for distributed memory parallel programming.",
        "Work stealing allows idle threads to take tasks from busy threads.",
        "Barrier synchronization forces all threads to wait at a common point.",
        "False sharing degrades performance when threads share a cache line.",
        "A thread pool reuses worker threads to avoid repeated creation overhead.",
    ]),
    ("distributed-systems", [
        "Distributed systems coordinate multiple machines to work together.",
        "Consensus algorithms allow distributed nodes to agree on a value.",
        "Replication improves fault tolerance by copying data across nodes.",
        "Partitioning splits data across multiple nodes to improve scalability.",
        "MapReduce is a programming model for processing large datasets in parallel.",
        "Message passing is a common communication pattern in distributed systems.",
        "Load balancing distributes work evenly across available workers.",
        "Fault tolerance allows a system to continue operating when components fail.",
        "Eventual consistency allows temporary divergence between replicas.",
        "A coordinator process aggregates results from distributed workers.",
        "Heartbeat messages detect when a worker process has failed.",
        "Two-phase commit ensures all participants agree before a transaction commits.",
    ]),
]


def make_document(topic: str, sentences: list[str], n_sentences: int) -> str:
    chosen = random.sample(sentences, min(n_sentences, len(sentences)))
    title = topic.replace("-", " ").title()
    return f"# {title}\n\n" + "\n".join(chosen) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic corpus for dsearch")
    parser.add_argument("--output-dir", default="./data", help="Output directory")
    parser.add_argument("--count",      type=int, default=25, help="Number of documents")
    parser.add_argument("--seed",       type=int, default=42,  help="Random seed")
    parser.add_argument("--multiplier", type=int, default=1,   help="Repeat content to simulate larger docs")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    docs_per_topic = max(1, args.count // len(TOPICS))
    doc_num = 0

    for topic, sentences in TOPICS:
        for _ in range(docs_per_topic):
            n_sentences = random.randint(4, 8)
            content = make_document(topic, sentences, n_sentences)
            if args.multiplier > 1:
                content = content * args.multiplier
            filename = f"{topic}-{doc_num:03d}.txt"
            path     = os.path.join(args.output_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            doc_num += 1

    print(f"Generated {doc_num} documents in {args.output_dir}/")


if __name__ == "__main__":
    main()
