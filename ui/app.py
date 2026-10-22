#!/usr/bin/env python3
"""
dsearch Interactive Web UI Backend
Zero-external-dependency web server built with Python standard library.
Interfaces with compiled C++ binaries:
  - ./build/dsearch (Serial and OpenMP execution)
  - ./build/dsearch_mpi (Hybrid OpenMPI + OpenMP execution)
  - ./build/dsearch_compare (Live benchmark harness)
"""

import os
import sys
import json
import time
import re
import subprocess
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

PORT = 8080
BASE_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = BASE_DIR / "build"
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"
BENCH_DATA_DIR = BASE_DIR / "bench_compare_data"

OPENMPI_PATH = "/usr/lib64/openmpi/bin"
OPENMPI_LIB = "/usr/lib64/openmpi/lib"

def get_env():
    env = os.environ.copy()
    current_path = env.get("PATH", "")
    if OPENMPI_PATH not in current_path:
        env["PATH"] = f"{OPENMPI_PATH}:{current_path}"
    current_ld = env.get("LD_LIBRARY_PATH", "")
    if OPENMPI_LIB not in current_ld:
        env["LD_LIBRARY_PATH"] = f"{OPENMPI_LIB}:{current_ld}"
    return env

def get_snippet(file_path_str, query_terms=None):
    try:
        p_obj = Path(file_path_str)
        if p_obj.is_absolute():
            path = p_obj
        else:
            path = BASE_DIR / file_path_str.lstrip("./")
        if not path.exists():
            return "Document content unavailable."
        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = [line.strip() for line in text.split("\n") if line.strip() and not line.startswith("#")]
        content = " ".join(lines)
        return (content[:240] + "...") if len(content) > 240 else content
    except Exception as e:
        return f"Error reading document: {e}"

def execute_search(query, mode="serial", top_k=5, corpus="./data", omp_threads=4, mpi_ranks=4):
    env = get_env()
    t_start = time.perf_counter()
    results = []
    output_raw = ""
    backend_time_ms = 0.0

    # Expand tilde ~ to full home path (e.g. ~/Documents -> /home/chetan/Documents)
    # C++ std::filesystem does not natively expand ~
    corpus_expanded = os.path.expanduser(corpus)

    if mode == "serial":
        cmd = [
            str(BUILD_DIR / "dsearch"),
            "search", query,
            "--corpus", corpus_expanded,
            "--top-k", str(top_k),
            "--threads", "1"
        ]
        proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True, env=env)
        output_raw = proc.stdout + proc.stderr
        t_end = time.perf_counter()
        backend_time_ms = (t_end - t_start) * 1000.0

        for line in proc.stdout.splitlines():
            m = re.match(r"\s*(\d+)\.\s+(\S+)\s+score=([0-9.]+)", line)
            if m:
                rank_num = int(m.group(1))
                doc_path = m.group(2)
                score = float(m.group(3))
                results.append({
                    "rank": rank_num,
                    "path": doc_path,
                    "name": Path(doc_path).name,
                    "score": score,
                    "snippet": get_snippet(doc_path, query.split())
                })

    elif mode == "openmp":
        cmd = [
            str(BUILD_DIR / "dsearch"),
            "search", query,
            "--corpus", corpus_expanded,
            "--top-k", str(top_k),
            "--threads", str(omp_threads)
        ]
        env["OMP_NUM_THREADS"] = str(omp_threads)
        proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True, env=env)
        output_raw = proc.stdout + proc.stderr
        t_end = time.perf_counter()
        backend_time_ms = (t_end - t_start) * 1000.0

        for line in proc.stdout.splitlines():
            m = re.match(r"\s*(\d+)\.\s+(\S+)\s+score=([0-9.]+)", line)
            if m:
                rank_num = int(m.group(1))
                doc_path = m.group(2)
                score = float(m.group(3))
                results.append({
                    "rank": rank_num,
                    "path": doc_path,
                    "name": Path(doc_path).name,
                    "score": score,
                    "snippet": get_snippet(doc_path, query.split())
                })

    elif mode == "hybrid_mpi":
        mpirun_bin = f"{OPENMPI_PATH}/mpirun" if os.path.exists(f"{OPENMPI_PATH}/mpirun") else "mpirun"
        cmd = [
            mpirun_bin,
            "-np", str(mpi_ranks),
            "-mca", "btl", "self,vader",
            str(BUILD_DIR / "dsearch_mpi"),
            "-c", corpus_expanded,
            "-q", query,
            "-k", str(top_k),
            "-t", str(omp_threads)
        ]
        proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True, env=env)
        output_raw = proc.stdout + proc.stderr
        t_end = time.perf_counter()
        
        # Check if C++ output has exact measured query execution time:
        # e.g.: [MPI] Query: "database" executed in 0.4334 ms
        cpp_query_ms = None
        for line in proc.stdout.splitlines():
            qm = re.search(r"executed in\s+([0-9.]+)\s+ms", line)
            if qm:
                cpp_query_ms = float(qm.group(1))
                break

        backend_time_ms = cpp_query_ms if cpp_query_ms is not None else ((t_end - t_start) * 1000.0)

        # Parse MPI output:
        # 1. [0.3486] ./data/database-000.txt (doc_id=0)
        for line in proc.stdout.splitlines():
            m = re.match(r"\s*(\d+)\.\s+\[([0-9.]+)\]\s+(\S+)", line)
            if m:
                rank_num = int(m.group(1))
                score = float(m.group(2))
                doc_path = m.group(3)
                results.append({
                    "rank": rank_num,
                    "path": doc_path,
                    "name": Path(doc_path).name,
                    "score": score,
                    "snippet": get_snippet(doc_path, query.split())
                })

    return {
        "mode": mode,
        "query": query,
        "results": results,
        "count": len(results),
        "latency_ms": round(backend_time_ms, 2),
        "omp_threads": omp_threads,
        "mpi_ranks": mpi_ranks if mode == "hybrid_mpi" else 1,
        "raw_log": output_raw
    }

def get_stats(corpus_dir="./data"):
    env = get_env()
    expanded_dir = os.path.expanduser(corpus_dir)
    cmd = [str(BUILD_DIR / "dsearch"), "stats", expanded_dir]
    proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True, env=env)
    
    docs = 0
    vocab = 0
    tokens = 0
    avg_tokens = 0.0

    for line in proc.stdout.splitlines():
        if "Documents" in line:
            parts = line.split(":")
            if len(parts) > 1: docs = int(parts[1].strip())
        elif "Vocabulary" in line:
            m = re.search(r"(\d+)", line)
            if m: vocab = int(m.group(1))
        elif "Total tokens" in line:
            parts = line.split(":")
            if len(parts) > 1: tokens = int(parts[1].strip())
        elif "Avg tokens/doc" in line:
            parts = line.split(":")
            if len(parts) > 1: avg_tokens = float(parts[1].strip())

    return {
        "documents": docs,
        "vocabulary": vocab,
        "total_tokens": tokens,
        "avg_tokens_per_doc": avg_tokens,
        "corpus_dir": corpus_dir
    }

def run_benchmark(corpus_dir=None):
    env = get_env()
    mpirun_bin = f"{OPENMPI_PATH}/mpirun" if os.path.exists(f"{OPENMPI_PATH}/mpirun") else "mpirun"
    
    if corpus_dir:
        corpus_target = os.path.expanduser(corpus_dir)
    elif BENCH_DATA_DIR.exists():
        corpus_target = str(BENCH_DATA_DIR)
    else:
        corpus_target = str(DATA_DIR)
    
    cmd = [
        mpirun_bin,
        "-np", "4",
        "-mca", "btl", "self,vader",
        str(BUILD_DIR / "dsearch_compare"),
        corpus_target
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True, env=env)
    t1 = time.perf_counter()

    parsed_data = []
    for line in proc.stdout.splitlines():
        # Examples:
        # Serial (1 Process, 1 Thread)    229.13              1.00x (baseline)  0.1678              1.00x (baseline)
        # Parallel (OpenMP 6 Threads)     187.22              1.22x             0.4036              0.41x
        # Hybrid (MPI 4 Ranks + OpenMP)   231.38              0.99x             0.1817              0.92x
        parts = line.strip().split()
        if len(parts) >= 5 and parts[0] in ["Serial", "Parallel", "Hybrid"]:
            mode_name = parts[0]
            # Find numbers from end or scan
            # We know the last columns are: idx_time, idx_speedup, query_time, query_speedup
            nums = re.findall(r"([0-9]+\.[0-9]+(?:x)?)", line)
            if len(nums) >= 4:
                # Mode label is everything before the first number
                first_num_pos = re.search(r"[0-9]+\.[0-9]+", line).start()
                label = line[:first_num_pos].strip()
                idx_time = float(nums[0].rstrip("x"))
                idx_speed = nums[1] if "x" in nums[1] else f"{nums[1]}x"
                q_time = float(nums[2].rstrip("x"))
                q_speed = nums[3] if "x" in nums[3] else f"{nums[3]}x"
                parsed_data.append({
                    "mode": label,
                    "idx_time_ms": idx_time,
                    "idx_speedup": idx_speed,
                    "query_latency_ms": q_time,
                    "query_speedup": q_speed
                })

    return {
        "benchmark_rows": parsed_data,
        "total_time_ms": round((t1 - t0) * 1000, 2),
        "raw_output": proc.stdout + proc.stderr
    }

class RequestHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            html_path = BASE_DIR / "ui" / "index.html"
            if html_path.exists():
                content = html_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, "index.html not found")
            return

        elif path == "/api/stats":
            query_params = urllib.parse.parse_qs(parsed.query)
            corpus = query_params.get("corpus", ["./data"])[0]
            data = get_stats(corpus)
            body = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        elif path == "/api/document":
            query_params = urllib.parse.parse_qs(parsed.query)
            doc_path = query_params.get("path", [""])[0]
            try:
                p_obj = Path(doc_path)
                if p_obj.is_absolute():
                    full_path = p_obj.resolve()
                else:
                    full_path = (BASE_DIR / doc_path.lstrip("./")).resolve()
                text = full_path.read_text(encoding="utf-8", errors="ignore")
                body = json.dumps({"path": str(full_path), "content": text}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self.send_error(500, str(e))
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b"{}"

        try:
            req_json = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            req_json = {}

        if path == "/api/search":
            query = req_json.get("query", "database")
            mode = req_json.get("mode", "serial")
            top_k = int(req_json.get("top_k", 5))
            corpus = req_json.get("corpus", "./data")
            omp_threads = int(req_json.get("omp_threads", 4))
            mpi_ranks = int(req_json.get("mpi_ranks", 4))

            resp = execute_search(
                query=query,
                mode=mode,
                top_k=top_k,
                corpus=corpus,
                omp_threads=omp_threads,
                mpi_ranks=mpi_ranks
            )

            resp_bytes = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_bytes)))
            self.end_headers()
            self.wfile.write(resp_bytes)
            return

        elif path == "/api/benchmark":
            corpus = req_json.get("corpus", None)
            resp = run_benchmark(corpus_dir=corpus)
            resp_bytes = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_bytes)))
            self.end_headers()
            self.wfile.write(resp_bytes)
            return

        self.send_error(404, "Not Found")

def main():
    server_address = ("0.0.0.0", PORT)
    httpd = HTTPServer(server_address, RequestHandler)
    print("=" * 65)
    print(f"🚀 dsearch Interactive Web UI server running at:")
    print(f"   http://localhost:{PORT}")
    print(f"   http://127.0.0.1:{PORT}")
    print("=" * 65)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dsearch UI server.")
        httpd.server_close()

if __name__ == "__main__":
    main()
