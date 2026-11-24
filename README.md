# Micro Metal

| M5 Pro operation | Tile (M × N × K) | Dependent latency (ns/call) | Throughput (TFLOP/s) |
|---|---:|---:|---:|
| `matmul2d` BF16 → FP32 | 16 × 32 × 16 | 43.3 | 31.46 |
| `matmul2d` BF16 → FP32 | 32 × 32 × 16 | 180.2 | 29.61 |
| `matmul2d` BF16 → FP32 | 32 × 32 × 32 | 258.4 | 31.45 |
