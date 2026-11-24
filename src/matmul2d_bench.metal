#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#if INPUT_FORMAT == 0
using Element = bfloat;
using Storage = bfloat;
#elif INPUT_FORMAT == 1
using Element = half;
using Storage = half;
#elif INPUT_FORMAT == 2
using Element = metal_fp8_e4m3_format;
using Storage = uchar;
#else
using Element = metal_fp8_e5m2_format;
using Storage = uchar;
#endif

#if SIMDGROUPS > 0
kernel void measure_gemm(device Storage* a [[buffer(0)]],
                         device Storage* b [[buffer(1)]],
                         device float* c [[buffer(2)]],
                         constant int& n [[buffer(3)]],
                         uint2 group [[threadgroup_position_in_grid]]) {
  using Input = tensor<device Element, dextents<int, 2>, tensor_inline>;
  Input lhs(a, dextents<int, 2>(n, n));
  Input rhs(b, dextents<int, 2>(n, n));
  auto dst = tensor(c, dextents<int, 2>(n, n), array<int, 2>{1, n});

  constexpr auto descriptor = matmul2d_descriptor(TILE_M, TILE_N, dynamic_length_v<int>,
                                                   false, false, false,
                                                   matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<SIMDGROUPS>> op;
  auto left = lhs.slice(0, group.y * TILE_M);
  auto right = rhs.slice(group.x * TILE_N, 0);
  auto output = dst.slice(group.x * TILE_N, group.y * TILE_M);
  op.run(left, right, output);
}
#else
kernel void measure_matmul2d(device Storage* a [[buffer(0)]],
                            device Storage* b [[buffer(1)]],
                            device float* c [[buffer(2)]],
                            constant uint& repeats [[buffer(3)]],
                            uint group [[threadgroup_position_in_grid]],
                            uint simdgroup [[simdgroup_index_in_threadgroup]],
                            uint simdgroupsPerThreadgroup [[simdgroups_per_threadgroup]]) {
  constexpr int m = TILE_M, n = TILE_N, k = TILE_K;
  constexpr auto descriptor = matmul2d_descriptor(m, n, k, false, false, false,
                                                   matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroup> op;
#if INPUT_FORMAT == 0
  auto lhs = tensor(a, extents<int, k, m>{}, array<int, 2>{1, k});
  auto rhs = tensor(b, extents<int, n, k>{}, array<int, 2>{1, n});
  auto left = op.get_left_input_cooperative_tensor<Element, Element, float>();
  auto right = op.get_right_input_cooperative_tensor<Element, Element, float>();
#else
  auto left = tensor<device Element, extents<int, k, m>, tensor_inline>(
      a, extents<int, k, m>{}, array<int, 2>{1, BUFFER_STRIDE});
  auto right = tensor<device Element, extents<int, n, k>, tensor_inline>(
      b, extents<int, n, k>{}, array<int, 2>{1, BUFFER_STRIDE});
#endif
  using Left = remove_addrspace_t<decltype(left)>;
  using Right = remove_addrspace_t<decltype(right)>;
  auto acc = op.get_destination_cooperative_tensor<Left, Right, float>();
#if INPUT_FORMAT == 0
  left.load(lhs);
  right.load(rhs);
#endif
  for (int i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;

  #pragma clang loop unroll_count(4)
  for (uint i = 0; i < repeats; ++i) op.run(left, right, acc);

  const uint tile = group * simdgroupsPerThreadgroup + simdgroup;
  auto dst = tensor(c + tile * (m * n), extents<int, n, m>{}, array<int, 2>{1, n});
  acc.store(dst);
}
#endif
