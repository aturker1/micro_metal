#include <metal_stdlib>
using namespace metal;

#if USE_FP16
using Scalar = half;
#else
using Scalar = float;
#endif

kernel void alu_bench(
    device Scalar* output [[buffer(0)]],
    constant uint& rounds [[buffer(1)]],
    constant Scalar& scale [[buffer(2)]],
    constant Scalar& step [[buffer(3)]],
    uint lane [[thread_position_in_grid]]) {
  Scalar values[CHAINS];
  #pragma clang loop unroll(full)
  for (uint chain = 0; chain < CHAINS; ++chain) {
    values[chain] = Scalar(float((lane & 255u) + chain * 17u) / 256.0f);
  }

  for (uint round = 0; round < rounds; ++round) {
    #pragma clang loop unroll(full)
    for (uint op = 0; op < OPS_PER_ROUND; ++op) {
      #pragma clang loop unroll(full)
      for (uint chain = 0; chain < CHAINS; ++chain) {
#if ALU_ADD
        // Alternate the sign to keep the FP16 chain in a range with room to add.
        values[chain] += (op & 1u) ? -step : step;
#elif ALU_FMA
        values[chain] = fma(values[chain], scale, step);
#elif USE_FP16
        // Negate the input to keep the dependent chain in range.
        values[chain] = exp2(-values[chain]);
#else
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
