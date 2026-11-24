# Micro Metal

| M5 Pro FP32 ALU (one op per lane) | Dependent latency (ns/op) | Est. ops/clock (20-core GPU at 1.62 GHz) | Est. ops/clock per core |
|---|---:|---:|---:|
| Add | 1.33 | 2,507 | 125.4 |
| FMA | 1.33 | 2,450 | 122.5 |

| M5 Pro `matmul2d` BF16 → FP32 (M × N × K) | Dependent latency (ns/tile) | Est. tiles/clock (20-core GPU at 1.62 GHz) | Est. tiles/clock per core |
|---|---:|---:|---:|
| 16 × 32 × 16 | 43.2 | 1.185 | 0.0593 |
| 32 × 32 × 16 | 180.2 | 0.560 | 0.0280 |
| 32 × 32 × 32 | 258.8 | 0.296 | 0.0148 |
