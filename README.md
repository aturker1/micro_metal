# Micro Metal

| M5 Pro FP32 ALU (one op per lane) | Dependent latency (ns/op) | Est. ops/clock (20-core GPU at 1.62 GHz) | Est. ops/clock per core |
|---|---:|---:|---:|
| Add | 1.33 | 2,505 | 125.2 |
| FMA | 1.33 | 2,447 | 122.4 |
| `fast::exp2(-x)` | 3.16 | 1,230 | 61.5 |

| M5 Pro `matmul2d` inputs → FP32 | Tile (M × N × K) | Operand source | Dependent latency (ns/tile) | Est. tiles/clock (20-core GPU at 1.62 GHz) | Est. tiles/clock per core |
|---|---:|---|---:|---:|---:|
| BF16 | 16 × 32 × 16 | Registers | 43.2 | 1.185 | 0.0593 |
| BF16 | 32 × 32 × 16 | Registers | 183.5 | 0.561 | 0.0281 |
| BF16 | 32 × 32 × 32 | Registers | 258.7 | 0.296 | 0.0148 |
| FP16 | 16 × 32 × 32 | Reused GPU buffers | 130.3 | 0.607 | 0.0304 |
| FP8 E4M3 | 16 × 32 × 32 | Reused GPU buffers | 156.2 | 0.601 | 0.0301 |
| FP16 | 32 × 32 × 32 | Reused GPU buffers | 313.0 | 0.300 | 0.0150 |
| FP8 E4M3 | 32 × 32 × 32 | Reused GPU buffers | 322.9 | 0.296 | 0.0148 |
| FP16 | 32 × 32 × 64 | Reused GPU buffers | 544.4 | 0.151 | 0.00755 |
| FP8 E4M3 | 32 × 32 × 64 | Reused GPU buffers | 612.8 | 0.150 | 0.00750 |
