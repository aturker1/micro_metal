#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

kernel void measure_matmul2d(device bfloat* a [[buffer(0)]],
                            device bfloat* b [[buffer(1)]],
                            device float* c [[buffer(2)]],
                            constant uint& repeats [[buffer(3)]],
                            uint group [[threadgroup_position_in_grid]],
                            uint simdgroup [[simdgroup_index_in_threadgroup]],
                            uint simdgroupsPerThreadgroup [[simdgroups_per_threadgroup]]) {
  constexpr int m = TILE_M, n = TILE_N, k = TILE_K;
  constexpr auto descriptor = matmul2d_descriptor(m, n, k, false, false, false,
                                                   matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroup> op;
  auto lhs = tensor(a, extents<int, k, m>{}, array<int, 2>{1, k});
  auto rhs = tensor(b, extents<int, n, k>{}, array<int, 2>{1, n});
  auto left = op.get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto right = op.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  using Left = remove_addrspace_t<decltype(left)>;
  using Right = remove_addrspace_t<decltype(right)>;
  auto acc = op.get_destination_cooperative_tensor<Left, Right, float>();
  left.load(lhs);
  right.load(rhs);
  for (int i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;

  #pragma clang loop unroll_count(4)
  for (uint i = 0; i < repeats; ++i) op.run(left, right, acc);

  const uint tile = group * simdgroupsPerThreadgroup + simdgroup;
  auto dst = tensor(c + tile * (m * n), extents<int, n, m>{}, array<int, 2>{1, n});
  acc.store(dst);
}
