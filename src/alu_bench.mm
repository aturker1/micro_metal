#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <utility>

#include "alu_shader.h"
#include "gpu_clock.h"

namespace {
constexpr uint32_t kOpsPerRound = 64;
constexpr uint32_t kThroughputThreads = 65536;
constexpr uint32_t kThroughputChains = 8;
template <typename Scalar> constexpr Scalar kScale = -0.99951171875f;
template <typename Scalar> constexpr Scalar kStep = 1.0f / 512.0f;

enum class Operation { Add, FMA, Exp2 };
constexpr std::array kOperations{
    std::pair{Operation::Add, "Add"},
    std::pair{Operation::FMA, "FMA"},
    std::pair{Operation::Exp2, "Exp2"},
};

id<MTLComputePipelineState> makePipeline(id<MTLDevice> device, Operation operation, uint32_t chains, bool fp16) {
  MTLCompileOptions* options = [MTLCompileOptions new];
  options.mathMode = MTLMathModeSafe;
  options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsFast;
  options.preprocessorMacros = @{
    @"ALU_ADD": @(operation == Operation::Add ? 1 : 0),
    @"ALU_FMA": @(operation == Operation::FMA ? 1 : 0),
    @"USE_FP16": @(fp16 ? 1 : 0),
    @"CHAINS": @(chains), @"OPS_PER_ROUND": @(kOpsPerRound)
  };
  NSError* error = nil;
  id<MTLLibrary> library = [device newLibraryWithSource:@(aluShader) options:options error:&error];
  if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
  id<MTLFunction> function = [library newFunctionWithName:@"alu_bench"];
  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
  if (!pipeline) throw std::runtime_error(error.localizedDescription.UTF8String);
  return pipeline;
}

template <typename Scalar>
double timeDispatch(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                    id<MTLBuffer> output, uint32_t threads, uint32_t rounds) {
  @autoreleasepool {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:output offset:0 atIndex:0];
    [encoder setBytes:&rounds length:sizeof(rounds) atIndex:1];
    [encoder setBytes:&kScale<Scalar> length:sizeof(Scalar) atIndex:2];
    [encoder setBytes:&kStep<Scalar> length:sizeof(Scalar) atIndex:3];
    const NSUInteger groupSize = std::min({
        NSUInteger(threads), 8 * pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup});
    [encoder dispatchThreads:MTLSizeMake(threads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(groupSize, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
      throw std::runtime_error(command.error.localizedDescription.UTF8String);
    }
    return command.GPUEndTime - command.GPUStartTime;
  }
}

template <typename Scalar>
void verify(id<MTLBuffer> output, Operation operation, uint32_t threads, uint32_t chains, uint32_t rounds) {
  const auto* values = static_cast<const Scalar*>(output.contents);
  const float exp2Tolerance = sizeof(Scalar) == 2 ? 0.001f : 1e-5f;
  const float tolerance = operation == Operation::Exp2 ? exp2Tolerance : 0.0f;
  const float scale = float(kScale<Scalar>);
  const float step = float(kStep<Scalar>);
  for (uint32_t lane : {0u, threads / 3, threads - 1}) {
    for (uint32_t chain = 0; chain < chains; ++chain) {
      Scalar expected = Scalar(float((lane & 255u) + chain * 17u) / 256.0f);
      for (uint32_t op = 0; op < rounds * kOpsPerRound; ++op) {
        switch (operation) {
          case Operation::Add: expected = Scalar(float(expected) + ((op & 1u) ? -step : step)); break;
          case Operation::FMA: expected = Scalar(std::fma(float(expected), scale, step)); break;
          case Operation::Exp2: expected = Scalar(std::exp2(-float(expected))); break;
        }
      }
      if (!(std::abs(float(values[lane * chains + chain]) - float(expected)) <= tolerance)) {
        throw std::runtime_error("GPU result disagrees with the CPU reference");
      }
    }
  }
}

template <typename Scalar>
double secondsPerStep(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                      id<MTLBuffer> output, Operation operation, uint32_t threads, uint32_t chains,
                      uint32_t shortRounds, uint32_t longRounds) {
  std::array<double, 31> differences;
  for (size_t i = 0; i < differences.size(); ++i) {
    double shorter, longer;
    if (i % 2 == 0) {
      shorter = timeDispatch<Scalar>(queue, pipeline, output, threads, shortRounds);
      longer = timeDispatch<Scalar>(queue, pipeline, output, threads, longRounds);
    } else {
      longer = timeDispatch<Scalar>(queue, pipeline, output, threads, longRounds);
      shorter = timeDispatch<Scalar>(queue, pipeline, output, threads, shortRounds);
    }
    differences[i] = longer - shorter;
  }
  std::sort(differences.begin(), differences.end());
  const double seconds = differences[differences.size() / 2];
  if (seconds <= 0) throw std::runtime_error("GPU timings could not resolve the added operations");

  timeDispatch<Scalar>(queue, pipeline, output, threads, longRounds);
  verify<Scalar>(output, operation, threads, chains, longRounds);
  return seconds / ((longRounds - shortRounds) * kOpsPerRound);
}

template <typename Scalar>
void measure(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLBuffer> output, double clockHz) {
  constexpr bool fp16 = sizeof(Scalar) == 2;
  for (const auto& [operation, name] : kOperations) {
    id<MTLComputePipelineState> dependent = makePipeline(device, operation, 1, fp16);
    id<MTLComputePipelineState> independent = makePipeline(device, operation, kThroughputChains, fp16);
    const auto simdWidth = static_cast<uint32_t>(dependent.threadExecutionWidth);
    for (int warmup = 0; warmup < 20; ++warmup) {
      timeDispatch<Scalar>(queue, independent, output, kThroughputThreads, 512);
    }
    const double throughputSeconds = secondsPerStep<Scalar>(
        queue, independent, output, operation, kThroughputThreads, kThroughputChains, 128, 256);
    const double latencySeconds = secondsPerStep<Scalar>(
        queue, dependent, output, operation, simdWidth, 1, 128, 512);
    const double opsPerStep = double(kThroughputThreads) * kThroughputChains;
    std::printf("%-10s %-9s %19.3f %21.1f\n", name, fp16 ? "FP16" : "FP32",
                latencySeconds * 1e9, opsPerStep / (throughputSeconds * clockHz));
  }
}
}  // namespace

int main(int argc, char* argv[]) {
  @autoreleasepool {
    try {
      const double clockHz = gpuClockHz(argc, argv);
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (!device) throw std::runtime_error("No Metal device");
      id<MTLCommandQueue> queue = [device newCommandQueue];
      id<MTLBuffer> output =
          [device newBufferWithLength:kThroughputThreads * kThroughputChains * sizeof(float)
                              options:MTLResourceStorageModeShared];
      if (!queue || !output) throw std::runtime_error("Could not create Metal benchmark resources");

      std::printf("%s | ALU | normalized to %g MHz\n", device.name.UTF8String, clockHz / 1e6);
      std::printf("%-10s %-9s %19s %21s\n", "Operation", "Precision", "Dependent ns/op", "Ops/device clock");
      measure<float>(device, queue, output, clockHz);
      measure<_Float16>(device, queue, output, clockHz);
    } catch (const std::exception& error) {
      std::fprintf(stderr, "%s\n", error.what());
      return 1;
    }
  }
}
