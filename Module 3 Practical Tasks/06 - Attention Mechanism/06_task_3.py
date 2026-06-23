# Task 3: KV Cache Benchmark Simulation
# Write a simulation that compares text generation with and without KV cache. 
# Simulate generating 100 tokens sequentially. 
# For each step, record: (a) how many K/V vectors are computed without cache, (b) how many with cache, and (c) the cumulative total. 
# Plot or print the cumulative computation over time for both approaches. Calculate the total speedup ratio. 
# Then estimate the memory cost of the KV cache for generating 4096 tokens with a model that has d_model=4096 and 32 layers (float16 precision). 
# Present your results in a clear summary.

import numpy as np

def run_kv_cache_simulation(total_generated_tokens=100):
    print("=" * 65)

    ops_without_cache = []
    ops_with_cache = []
    
    for step in range(1, total_generated_tokens + 1):
        ops_without_cache.append(step)
        
        ops_with_cache.append(1)
        
    cumulative_without = np.cumsum(ops_without_cache)
    cumulative_with = np.cumsum(ops_with_cache)
    
    total_ops_without = cumulative_without[-1]
    total_ops_with = cumulative_with[-1]
    speedup_ratio = total_ops_without / total_ops_with
    
    print(f"{'Step':<10}{'Without Cache (Step)':<22}{'With Cache (Step)':<20}{'Cumulative Naive':<18}{'Cumulative Cached'}")
    print("-" * 85)
    for idx in [0, 9, 24, 49, 74, 99]: 
        print(f"Step {idx+1:<5}{ops_without_cache[idx]:<22}{ops_with_cache[idx]:<20}{cumulative_without[idx]:<18}{cumulative_with[idx]}")
    print("-" * 85)
    
    return total_ops_without, total_ops_with, speedup_ratio

total_naive, total_cached, speedup = run_kv_cache_simulation()