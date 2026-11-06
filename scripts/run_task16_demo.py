"""
run_task16_demo.py - Demonstrates in-memory response caching with timing and call counter benchmarks (Task 16).
Outputs to transcripts/task16_cache_benchmark.txt.
"""

import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag.grounded_gen import (
    generate_grounded_response,
    clear_cache,
    get_cache_stats,
    normalize_query
)

def main():
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("TASK 16: IN-MEMORY RESPONSE CACHING BENCHMARK (NORMALIZED QUERY TEXT)")
    output_lines.append("=" * 80)
    output_lines.append("Cache Design:")
    output_lines.append("  - In-memory dictionary keyed by normalized query string (lowercase, stripped punctuation).")
    output_lines.append("  - Eliminates redundant ChromaDB vector similarity computations and generation latency.")
    output_lines.append("  - Tracks cache hits, misses, and execution latency.")
    output_lines.append("-" * 80)

    # Reset cache before benchmark
    clear_cache()
    
    # Test Query 1: Cold Call
    q1_raw = "What is the cancellation and rescheduling window for appointments?"
    output_lines.append("\n[TRIAL 1: COLD QUERY EXECUTION (CACHE MISS)]")
    output_lines.append(f"Input Query: '{q1_raw}'")
    output_lines.append(f"Normalized Key: '{normalize_query(q1_raw)}'")
    
    start_1 = time.perf_counter()
    res_1 = generate_grounded_response(q1_raw, use_cache=True)
    latency_1 = (time.perf_counter() - start_1) * 1000
    
    output_lines.append(f"Execution Latency: {latency_1:.2f} ms")
    output_lines.append(f"Cache Hit Flag: {res_1['cache_hit']}")
    output_lines.append(f"Cache Stats After Trial 1: {get_cache_stats()}")
    assert res_1["cache_hit"] is False, "Trial 1 must be a cache miss"

    output_lines.append("\n" + "-" * 80)

    # Test Query 2: Repeated Query with variations in whitespace, case, and punctuation
    q2_variant = "  WHAT is the cancellation and rescheduling window for appointments???  "
    output_lines.append("\n[TRIAL 2: WARM REPEATED QUERY EXECUTION (CACHE HIT)]")
    output_lines.append(f"Input Query: '{q2_variant}'")
    output_lines.append(f"Normalized Key: '{normalize_query(q2_variant)}'")
    
    start_2 = time.perf_counter()
    res_2 = generate_grounded_response(q2_variant, use_cache=True)
    latency_2 = (time.perf_counter() - start_2) * 1000
    
    output_lines.append(f"Execution Latency: {latency_2:.2f} ms")
    output_lines.append(f"Cache Hit Flag: {res_2['cache_hit']}")
    output_lines.append(f"Cache Stats After Trial 2: {get_cache_stats()}")
    assert res_2["cache_hit"] is True, "Trial 2 must be a cache hit"

    output_lines.append("\n" + "-" * 80)

    # Test Query 3: Second Distinct Policy Query
    q3_raw = "How do cashless insurance claims work for outpatient visits?"
    output_lines.append("\n[TRIAL 3: SECOND DISTINCT QUERY - COLD CALL]")
    output_lines.append(f"Input Query: '{q3_raw}'")
    
    start_3 = time.perf_counter()
    res_3 = generate_grounded_response(q3_raw, use_cache=True)
    latency_3 = (time.perf_counter() - start_3) * 1000
    output_lines.append(f"Execution Latency: {latency_3:.2f} ms (Cache Hit: {res_3['cache_hit']})")

    # Repeat Query 3: Warm Call
    start_4 = time.perf_counter()
    res_4 = generate_grounded_response(q3_raw, use_cache=True)
    latency_4 = (time.perf_counter() - start_4) * 1000
    output_lines.append(f"Repeated Execution Latency: {latency_4:.2f} ms (Cache Hit: {res_4['cache_hit']})")

    output_lines.append("\n" + "=" * 80)
    output_lines.append("CACHE PERFORMANCE BENCHMARK SUMMARY:")
    output_lines.append(f"  - Total Pipeline Calls: {get_cache_stats()['total_calls']}")
    output_lines.append(f"  - Total Cache Hits: {get_cache_stats()['hits']}")
    output_lines.append(f"  - Total Cache Misses: {get_cache_stats()['misses']}")
    speedup = latency_1 / max(0.001, latency_2)
    output_lines.append(f"  - Cold Call Latency (Trial 1): {latency_1:.2f} ms")
    output_lines.append(f"  - Warm Cache Hit Latency (Trial 2): {latency_2:.2f} ms")
    output_lines.append(f"  - Measured Latency Reduction: {speedup:.1f}x speedup")
    output_lines.append("=" * 80)

    text = "\n".join(output_lines)
    with open("transcripts/task16_cache_benchmark.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Task 16 transcript written to transcripts/task16_cache_benchmark.txt")

if __name__ == "__main__":
    main()
