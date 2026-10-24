# Benchmark results: vectorised vs Python loops

| Function | Vectorized (s) | Loop (s) | Speedup |
|---|---|---|---|
| StandardScaler (20,000 x 5) | 0.04245 | 0.8926 | 21.03x |
| euclidean_distance (150 x 150 pairs) | 0.003185 | 0.562 | 176.48x |
| rank, average ties (1,500 values) | 0.002363 | 4.062 | 1719.50x |

## Environment

- Date: 2026-10-24
- Python: 3.14.6
- NumPy: 2.5.1
- OS: Windows-11-10.0.26300-SP0
- CPU: Intel64 Family 6 Model 158 Stepping 10, GenuineIntel (8 logical cores)
- Method: mean of 5 timed runs after 1 warm-up run; fixed random seed 0
