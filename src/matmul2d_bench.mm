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
#include <string_view>
#include <vector>

namespace {

struct Shape { int m, n, k; };
struct Operands { id<MTLBuffer> a, b; };
enum class Format { BF16, FP16, E4M3, E5M2 };
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
constexpr std::array registerTiles{Shape{16, 32, 16}, Shape{32, 32, 16}, Shape{32, 32, 32}};
constexpr std::array bufferTiles{Shape{16, 32, 32}, Shape{32, 32, 32}, Shape{32, 32, 64}};
constexpr std::array formatNames{"BF16", "FP16", "FP8 E4M3", "FP8 E5M2"};
constexpr int bufferStride = 128;
constexpr uint32_t simdWidth = 32;
constexpr size_t samplePairs = 31;
// Allow an older host SDK to build the runner. The Metal compiler comes from the OS.
constexpr auto metal41 = static_cast<MTLLanguageVersion>(MTLLanguageVersion4_0 + 1);

const char* formatName(Format format) { return formatNames[static_cast<size_t>(format)]; }

id<MTLComputePipelineState> pipelineFor(id<MTLDevice> device, NSString* source, Shape shape,
                                       Format format, uint32_t groups = 0) {
  NSError* error = nil;
  MTLCompileOptions* options = [MTLCompileOptions new];
  options.languageVersion = format == Format::BF16 ? MTLLanguageVersion4_0 : metal41;
  options.mathMode = MTLMathModeSafe;
  options.preprocessorMacros = @{
    @"TILE_M": @(shape.m), @"TILE_N": @(shape.n), @"TILE_K": @(shape.k),
    @"INPUT_FORMAT": @(static_cast<int>(format)), @"BUFFER_STRIDE": @(bufferStride),
    @"SIMDGROUPS": @(groups)
  };
  id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
  if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
  id<MTLFunction> function = [library newFunctionWithName:groups ? @"measure_gemm" : @"measure_matmul2d"];
  if (!function) throw std::runtime_error("benchmark kernel is missing from the Metal library");
  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
  if (!pipeline) throw std::runtime_error(error.localizedDescription.UTF8String);
  const uint32_t groupsPerThreadgroup = groups ? groups : throughput.grid.simdgroupsPerThreadgroup;
  if (pipeline.threadExecutionWidth != simdWidth ||
      pipeline.maxTotalThreadsPerThreadgroup < groupsPerThreadgroup * simdWidth)
    throw std::runtime_error("GPU cannot execute the microbenchmark's SIMD-group layout");
  return pipeline;
}

// The test inputs are small dyadic numbers that fit exactly in every tested format.
float inputA(int row, int k) { return float((3 * row + 5 * k) % 7 - 3) / 8.0f; }
float inputB(int k, int col) { return float((2 * k + 3 * col) % 11 - 5) / 8.0f; }

id<MTLBuffer> allocate(id<MTLDevice> device, size_t bytes) {
  id<MTLBuffer> buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  if (!buffer) throw std::runtime_error("could not allocate a Metal buffer");
  return buffer;
}

void writeOperand(void* data, size_t index, Format format, float value) {
  if (format == Format::FP16) {
    static_cast<_Float16*>(data)[index] = value;
  } else if (format == Format::BF16) {
    static_cast<uint16_t*>(data)[index] = uint16_t(std::bit_cast<uint32_t>(value) >> 16);
  } else {
    const auto bits = std::bit_cast<uint32_t>(value);
    const int mantissaBits = format == Format::E4M3 ? 3 : 2;
    const int bias = format == Format::E4M3 ? 7 : 15;
    const int exponent = int((bits >> 23) & 255) - 127 + bias;
    const uint32_t mantissa = (bits & 0x7fffffu) >> (23 - mantissaBits);
    static_cast<uint8_t*>(data)[index] = value == 0 ? 0 :
        uint8_t(((bits >> 24) & 0x80u) | (uint32_t(exponent) << mantissaBits) | mantissa);
  }
}

Operands makeOperands(id<MTLDevice> device, Shape shape, Format format, int strideA, int strideB) {
  const size_t elementSize = format == Format::E4M3 || format == Format::E5M2 ? 1 : 2;
  auto a = allocate(device, size_t(shape.m) * strideA * elementSize);
  auto b = allocate(device, size_t(shape.k) * strideB * elementSize);
  void* lhs = a.contents;
  void* rhs = b.contents;
  for (int row = 0; row < shape.m; ++row)
    for (int k = 0; k < shape.k; ++k)
      writeOperand(lhs, size_t(row) * strideA + k, format, inputA(row, k));
  for (int k = 0; k < shape.k; ++k)
    for (int col = 0; col < shape.n; ++col)
      writeOperand(rhs, size_t(k) * strideB + col, format, inputB(k, col));
  return {a, b};
}

float referenceAt(int row, int col, int length) {
  float expected = 0;
  for (int k = 0; k < length; ++k) expected += inputA(row, k) * inputB(k, col);
  return expected;
}

void waitForCompletion(id<MTLCommandBuffer> command) {
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted)
    throw std::runtime_error(command.error ? command.error.localizedDescription.UTF8String : "Metal command failed");
}

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
             Format format, id<MTLCounterSampleBuffer> timestamps, double clockHz) {
  id<MTLComputePipelineState> pipeline = pipelineFor(device, source, shape, format);
  const size_t tileSize = size_t(shape.m) * shape.n;
  const bool registers = format == Format::BF16;
  const auto inputs = makeOperands(device, shape, format,
                                   registers ? shape.k : bufferStride, registers ? shape.n : bufferStride);
  id<MTLBuffer> c = allocate(device, tileSize * throughput.grid.totalSIMDGroups() * sizeof(float));
  auto* output = static_cast<const float*>(c.contents);
  std::vector<float> reference(tileSize);
  for (int row = 0; row < shape.m; ++row)
    for (int col = 0; col < shape.n; ++col)
      reference[row * shape.n + col] = referenceAt(row, col, shape.k);

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
      [encoder setBuffer:inputs.a offset:0 atIndex:0];
      [encoder setBuffer:inputs.b offset:0 atIndex:1];
      [encoder setBuffer:c offset:0 atIndex:2];
      [encoder setBytes:&repeats length:sizeof(repeats) atIndex:3];
      const auto groups = MTLSizeMake(grid.threadgroups, 1, 1);
      const auto threads = MTLSizeMake(grid.simdgroupsPerThreadgroup * simdWidth, 1, 1);
      for (uint32_t i = 0; i < grid.dispatches; ++i) [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
      [encoder endEncoding];
      waitForCompletion(command);

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
  std::printf("%-8s %2dx%2dx%-2d  %-14s %14.1f  %18.3f\n", formatName(format), shape.m, shape.n,
              shape.k, registers ? "Registers" : "Reused buffers", latencySeconds * 1e9, callsPerClock);
}

void verifyGemm(id<MTLBuffer> output, int n) {
  const auto* values = static_cast<const float*>(output.contents);
  auto check = [&](int row, int col) {
    const float expected = referenceAt(row, col, n);
    const float actual = values[size_t(row) * n + col];
    if (actual != expected) {
      std::fprintf(stderr, "Output at (%d, %d), got %g, expected %g\n", row, col, actual, expected);
      throw std::runtime_error("GEMM output validation failed");
    }
  };
  for (int row : {0, n - 1})
    for (int col : {0, n - 1}) check(row, col);
  uint32_t state = 6525;
  for (int sample = 0; sample < 96; ++sample) {
    state = state * 1664525u + 1013904223u;
    const int row = int((state >> 8) % n);
    state = state * 1664525u + 1013904223u;
    const int col = int((state >> 8) % n);
    check(row, col);
  }
}

void measureGemm(id<MTLDevice> device, id<MTLCommandQueue> queue, NSString* source) {
  constexpr int n = 8192;
  constexpr std::array formats{Format::FP16, Format::E4M3, Format::E5M2};
  constexpr unsigned warmups = 3, samples = 27;
  struct Tile { int m, n; uint32_t groups; };
  constexpr std::array tiles{Tile{64, 128, 4}, Tile{64, 64, 2}};
  std::array<Operands, formats.size()> inputs;
  for (size_t i = 0; i < formats.size(); ++i)
    inputs[i] = makeOperands(device, {n, n, n}, formats[i], n, n);
  id<MTLBuffer> output = allocate(device, size_t(n) * n * sizeof(float));

  std::puts("8192 x 8192 x 8192 | unscaled inputs, FP32 accumulation and output | Metal 4.1");
  std::printf("%u interleaved GPU timings per format\n", samples);
  std::puts("Inputs    Tile MxN  SIMD groups  Median GPU ms    TFLOP/s");
  for (const auto tile : tiles) {
    @autoreleasepool {
      std::array<id<MTLComputePipelineState>, formats.size()> pipelines;
      for (size_t i = 0; i < formats.size(); ++i)
        pipelines[i] = pipelineFor(device, source, {tile.m, tile.n, 0}, formats[i], tile.groups);

      auto dispatch = [&](size_t i) {
        @autoreleasepool {
          id<MTLCommandBuffer> command = [queue commandBuffer];
          id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
          if (!encoder) throw std::runtime_error("could not create a Metal compute encoder");
          [encoder setComputePipelineState:pipelines[i]];
          [encoder setBuffer:inputs[i].a offset:0 atIndex:0];
          [encoder setBuffer:inputs[i].b offset:0 atIndex:1];
          [encoder setBuffer:output offset:0 atIndex:2];
          [encoder setBytes:&n length:sizeof(n) atIndex:3];
          [encoder dispatchThreadgroups:MTLSizeMake(n / tile.n, n / tile.m, 1)
                   threadsPerThreadgroup:MTLSizeMake(tile.groups * simdWidth, 1, 1)];
          [encoder endEncoding];
          waitForCompletion(command);
          return command.GPUEndTime - command.GPUStartTime;
        }
      };

      for (size_t i = 0; i < warmups * formats.size(); ++i) dispatch(i % formats.size());
      std::array<std::array<double, samples>, formats.size()> timings;
      for (size_t sample = 0; sample < samples; ++sample) {
        for (size_t offset = 0; offset < formats.size(); ++offset) {
          const size_t i = (offset + sample) % formats.size();
          timings[i][sample] = dispatch(i);
          if (sample == samples - 1) verifyGemm(output, n);
        }
      }
      for (size_t i = 0; i < formats.size(); ++i) {
        auto& times = timings[i];
        std::sort(times.begin(), times.end());
        const double seconds = times[samples / 2];
        if (seconds <= 0) throw std::runtime_error("Metal returned invalid GPU timing");
        std::printf("%-8s %2dx%-3d %12u %14.3f %10.2f\n", formatName(formats[i]), tile.m,
                    tile.n, tile.groups, seconds * 1e3, 2.0 * n * n * n / seconds / 1e12);
      }
    }
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  @autoreleasepool {
    try {
      const bool gemm = argc == 2 && std::string_view(argv[1]) == "--gemm";
      const double clockHz = gemm ? 0 : gpuClockHz(argc, argv);
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (!device || ![device supportsFamily:MTLGPUFamilyApple10])
        throw std::runtime_error("the matmul2d microbenchmark requires an Apple10 or newer Metal GPU");
      id<MTLCommandQueue> queue = [device newCommandQueue];
      if (!queue) throw std::runtime_error("could not create a Metal command queue");
      NSString* source = [NSString stringWithUTF8String:matmul2dShader];
      if (!source) throw std::runtime_error("embedded Metal shader is not valid UTF-8");
      bool supportsMetal41 = false;
      if (@available(macOS 27.0, *)) supportsMetal41 = true;
      if (gemm) {
        if (!supportsMetal41) throw std::runtime_error("the FP8 GEMM benchmark requires macOS 27 or later");
        std::printf("%s | ", device.name.UTF8String);
        measureGemm(device, queue, source);
      } else {
        std::printf("%s | matmul2d -> FP32 | normalized to %g MHz\n", device.name.UTF8String, clockHz / 1e6);
        id<MTLCounterSampleBuffer> timestamps = timestampBufferFor(device);
        std::printf("Median of %zu paired GPU timings\n", samplePairs);
        std::puts("Inputs    Tile MxNxK  Operand source  Latency ns/tile  Tiles/device clock");
        for (const Shape shape : registerTiles)
          measure(device, queue, source, shape, Format::BF16, timestamps, clockHz);
        if (supportsMetal41) {
          for (const Shape shape : bufferTiles)
            for (Format format : {Format::FP16, Format::E4M3})
              measure(device, queue, source, shape, format, timestamps, clockHz);
        } else {
          std::puts("FP16 and FP8 reused-buffer tests require macOS 27 or later.");
        }
      }
      std::puts("All CPU reference checks passed.");
    } catch (const std::exception& error) {
      std::fprintf(stderr, "%s\n", error.what());
      return 1;
    }
  }
}
