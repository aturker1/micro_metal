#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "gpu_clock.h"
#include "matmul2d_shader.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

struct Shape { int m, n, k; };
struct Grid {
  uint32_t threadgroups, simdgroupsPerThreadgroup, dispatches;
  constexpr uint32_t totalSIMDGroups() const { return threadgroups * simdgroupsPerThreadgroup; }
};
struct Measurement {
  Grid grid;
  uint32_t shortRepeats, longRepeats, warmupRepeats;
};

constexpr Measurement latency{{1, 1, 64}, 52, 100, 1024};
constexpr Measurement throughput{{256, 8, 1}, 256, 512, 0};
constexpr std::array shapes{Shape{16, 32, 16}, Shape{32, 32, 16}, Shape{32, 32, 32}};
constexpr uint32_t simdWidth = 32;
constexpr size_t samplePairs = 31;

id<MTLComputePipelineState> pipelineFor(id<MTLDevice> device, NSString* source, Shape shape) {
  NSError* error = nil;
  MTLCompileOptions* options = [MTLCompileOptions new];
  options.languageVersion = MTLLanguageVersion4_0;
  options.mathMode = MTLMathModeSafe;
  options.preprocessorMacros = @{@"TILE_M": @(shape.m), @"TILE_N": @(shape.n), @"TILE_K": @(shape.k)};
  id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
  if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
  id<MTLFunction> function = [library newFunctionWithName:@"measure_matmul2d"];
  if (!function) throw std::runtime_error("measure_matmul2d is missing from the Metal library");
  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
  if (!pipeline) throw std::runtime_error(error.localizedDescription.UTF8String);
  if (pipeline.threadExecutionWidth != simdWidth ||
      pipeline.maxTotalThreadsPerThreadgroup < throughput.grid.simdgroupsPerThreadgroup * simdWidth)
    throw std::runtime_error("GPU cannot execute the microbenchmark's SIMD-group layout");
  return pipeline;
}

// The test inputs are small dyadic numbers that fit exactly in BF16.
uint16_t encodeBF16(float value) { return uint16_t(std::bit_cast<uint32_t>(value) >> 16); }
float decodeBF16(uint16_t bits) { return std::bit_cast<float>(uint32_t(bits) << 16); }
float inputA(int row, int k) { return float((3 * row + 5 * k) % 7 - 3) / 8.0f; }
float inputB(int k, int col) { return float((2 * k + 3 * col) % 11 - 5) / 8.0f; }

id<MTLCounterSampleBuffer> timestampBufferFor(id<MTLDevice> device) {
  if (![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) return nil;
  for (id<MTLCounterSet> set in device.counterSets) {
    if (![set.name isEqualToString:MTLCommonCounterSetTimestamp]) continue;
    MTLCounterSampleBufferDescriptor* descriptor = [MTLCounterSampleBufferDescriptor new];
    descriptor.counterSet = set;
    descriptor.sampleCount = 2;
    descriptor.storageMode = MTLStorageModeShared;
    return [device newCounterSampleBufferWithDescriptor:descriptor error:nullptr];
  }
  return nil;
}

double computePassSeconds(id<MTLCounterSampleBuffer> timestamps) {
  NSData* data = [timestamps resolveCounterRange:NSMakeRange(0, 2)];
  if (data.length != 2 * sizeof(MTLCounterResultTimestamp)) throw std::runtime_error("Metal timestamp samples are missing");
  auto* stamps = static_cast<const MTLCounterResultTimestamp*>(data.bytes);
  if (stamps[1].timestamp == MTLCounterErrorValue || stamps[1].timestamp <= stamps[0].timestamp)
    throw std::runtime_error("Metal returned invalid compute-pass timestamps");
  return double(stamps[1].timestamp - stamps[0].timestamp) * 1e-9;
}

void measure(id<MTLDevice> device, id<MTLCommandQueue> queue, NSString* source, Shape shape,
             id<MTLCounterSampleBuffer> timestamps, double clockHz) {
  id<MTLComputePipelineState> pipeline = pipelineFor(device, source, shape);
  const size_t tileSize = size_t(shape.m) * shape.n;
  auto allocate = [&](size_t bytes) {
    id<MTLBuffer> buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!buffer) throw std::runtime_error("could not allocate a Metal buffer");
    return buffer;
  };
  id<MTLBuffer> a = allocate(size_t(shape.m) * shape.k * sizeof(uint16_t));
  id<MTLBuffer> b = allocate(size_t(shape.k) * shape.n * sizeof(uint16_t));
  id<MTLBuffer> c = allocate(tileSize * throughput.grid.totalSIMDGroups() * sizeof(float));
  auto* lhs = static_cast<uint16_t*>(a.contents);
  auto* rhs = static_cast<uint16_t*>(b.contents);
  auto* output = static_cast<const float*>(c.contents);
  std::vector<float> reference(tileSize);
  for (int row = 0; row < shape.m; ++row)
    for (int k = 0; k < shape.k; ++k) lhs[row * shape.k + k] = encodeBF16(inputA(row, k));
  for (int k = 0; k < shape.k; ++k)
    for (int col = 0; col < shape.n; ++col) rhs[k * shape.n + col] = encodeBF16(inputB(k, col));
  for (int row = 0; row < shape.m; ++row)
    for (int col = 0; col < shape.n; ++col)
      for (int k = 0; k < shape.k; ++k)
        reference[row * shape.n + col] += decodeBF16(lhs[row * shape.k + k]) * decodeBF16(rhs[k * shape.n + col]);

  auto validate = [&](Grid grid, uint32_t repeats, bool checkAll) {
    const size_t tiles = grid.totalSIMDGroups();
    const size_t step = checkAll || tiles == 1 ? 1 : tiles - 1;
    for (size_t tile = 0; tile < tiles; tile += step) {
      for (size_t i = 0; i < tileSize; ++i) {
        const float actual = output[tile * tileSize + i];
        const float expected = reference[i] * float(repeats);
        if (actual != expected) {
          std::fprintf(stderr, "%dx%dx%d, repeat %u, tile %zu, index %zu, got %g, expected %g\n",
                       shape.m, shape.n, shape.k, repeats, tile, i, actual, expected);
          throw std::runtime_error("matmul2d output validation failed");
        }
      }
    }
  };

  MTLComputePassDescriptor* pass = [MTLComputePassDescriptor computePassDescriptor];
  pass.dispatchType = MTLDispatchTypeSerial;
  if (timestamps) {
    auto* attachment = pass.sampleBufferAttachments[0];
    attachment.sampleBuffer = timestamps;
    attachment.startOfEncoderSampleIndex = 0;
    attachment.endOfEncoderSampleIndex = 1;
  }
  auto dispatch = [&](Grid grid, uint32_t repeats, bool checkAll = false) {
    @autoreleasepool {
      const bool samplePass = timestamps && grid.dispatches > 1;
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = samplePass ? [command computeCommandEncoderWithDescriptor:pass] : [command computeCommandEncoder];
      if (!encoder) throw std::runtime_error("could not create a Metal compute encoder");
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:a offset:0 atIndex:0];
      [encoder setBuffer:b offset:0 atIndex:1];
      [encoder setBuffer:c offset:0 atIndex:2];
      [encoder setBytes:&repeats length:sizeof(repeats) atIndex:3];
      const auto groups = MTLSizeMake(grid.threadgroups, 1, 1);
      const auto threads = MTLSizeMake(grid.simdgroupsPerThreadgroup * simdWidth, 1, 1);
      for (uint32_t i = 0; i < grid.dispatches; ++i) [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error(command.error ? command.error.localizedDescription.UTF8String : "Metal command failed");

      validate(grid, repeats, checkAll);
      return samplePass ? computePassSeconds(timestamps) : command.GPUEndTime - command.GPUStartTime;
    }
  };

  dispatch(throughput.grid, 1, true);
  dispatch(throughput.grid, 2, true);
  // Sustain work long enough to warm up the GPU before taking short measurements.
  for (int i = 0; i < 8; ++i) dispatch(throughput.grid, 4096);

  auto timePerRepeat = [&](const Measurement& measurement) {
    const auto [grid, low, high, warmup] = measurement;
    const uint32_t extraRepeats = grid.dispatches * (high - low);
    std::array<double, samplePairs> intervals{};
    for (size_t i = 0; i < samplePairs; ++i) {
      if (warmup) dispatch(throughput.grid, warmup);
      // Alternate the order so the long dispatch doesn't always get the warmer GPU.
      double shortTime, longTime;
      if (i % 2 == 0) {
        shortTime = dispatch(grid, low);
        longTime = dispatch(grid, high);
      } else {
        longTime = dispatch(grid, high);
        shortTime = dispatch(grid, low);
      }
      intervals[i] = (longTime - shortTime) / extraRepeats;
    }
    std::sort(intervals.begin(), intervals.end());
    const double median = intervals[samplePairs / 2];
    if (median <= 0) throw std::runtime_error("GPU timing slope is zero or negative");
    return median;
  };

  const double deviceRepeatSeconds = timePerRepeat(throughput);
  const double latencySeconds = timePerRepeat(latency);
  const double callsPerClock = throughput.grid.totalSIMDGroups() / (deviceRepeatSeconds * clockHz);
  std::printf("%2dx%2dx%-2d  %14.1f  %18.3f\n", shape.m, shape.n, shape.k, latencySeconds * 1e9, callsPerClock);
}

}  // namespace

int main(int argc, char* argv[]) {
  @autoreleasepool {
    try {
      const double clockHz = gpuClockHz(argc, argv);
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (!device || ![device supportsFamily:MTLGPUFamilyApple10])
        throw std::runtime_error("the matmul2d microbenchmark requires an Apple10 or newer Metal GPU");
      id<MTLCommandQueue> queue = [device newCommandQueue];
      if (!queue) throw std::runtime_error("could not create a Metal command queue");
      NSString* source = [NSString stringWithUTF8String:matmul2dShader];
      if (!source) throw std::runtime_error("embedded Metal shader is not valid UTF-8");
      std::printf("%s | matmul2d BF16 x BF16 -> FP32 | normalized to %g MHz\n", device.name.UTF8String, clockHz / 1e6);
      id<MTLCounterSampleBuffer> timestamps = timestampBufferFor(device);
      std::printf("Median of %zu paired GPU timings\n", samplePairs);
      std::puts("Tile MxNxK  Latency ns/call  Calls/device clock");
      for (const Shape shape : shapes) measure(device, queue, source, shape, timestamps, clockHz);
      std::puts("All CPU reference checks passed.");
    } catch (const std::exception& error) {
      std::fprintf(stderr, "%s\n", error.what());
      return 1;
    }
  }
}
