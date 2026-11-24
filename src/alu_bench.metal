#include <metal_stdlib>
using namespace metal;

kernel void alu_bench(
    device float* output [[buffer(0)]],
    constant uint& rounds [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant float& step [[buffer(3)]],
    uint lane [[thread_position_in_grid]]) {
  float values[CHAINS];
  #pragma clang loop unroll(full)
  for (uint chain = 0; chain < CHAINS; ++chain) {
    values[chain] = float((lane & 255u) + chain * 17u) / 256.0f;
  }

  for (uint round = 0; round < rounds; ++round) {
    #pragma clang loop unroll(full)
    for (uint op = 0; op < OPS_PER_ROUND; ++op) {
      #pragma clang loop unroll(full)
      for (uint chain = 0; chain < CHAINS; ++chain) {
#if FUSED
        values[chain] = fma(values[chain], scale, step);
#else
        values[chain] += step;
#endif
      }
    }
  }

  #pragma clang loop unroll(full)
  for (uint chain = 0; chain < CHAINS; ++chain) {
    output[lane * CHAINS + chain] = values[chain];
  }
}
