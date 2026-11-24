# Micro Metal

| M5 Pro ALU (one op per lane) | Precision | Dependent latency (ns/op) | Est. ops/clock (20-core GPU at 1.62 GHz) | Est. ops/clock per core |
|---|---|---:|---:|---:|
| Add | FP32 | 1.33 | 2,496 | 124.8 |
| Add | FP16 | 1.33 | 4,855 | 242.8 |
| FMA | FP32 | 1.33 | 2,451 | 122.6 |
| FMA | FP16 | 1.33 | 4,814 | 240.7 |
| `fast::exp2(-x)` | FP32 | 3.16 | 1,229 | 61.5 |
| `exp2(-x)` | FP16 | 3.79 | 1,233 | 61.7 |

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
