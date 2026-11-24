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
#if ALU_ADD
        values[chain] += step;
#elif ALU_FMA
        values[chain] = fma(values[chain], scale, step);
#else
        // Negate the input to keep the dependent chain in range.
        values[chain] = fast::exp2(-values[chain]);
#endif
      }
    }
  }

  #pragma clang loop unroll(full)
  for (uint chain = 0; chain < CHAINS; ++chain) {
    output[lane * CHAINS + chain] = values[chain];
  }
}
