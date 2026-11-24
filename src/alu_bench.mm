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
constexpr float kScale = 0.9999237060546875f;
constexpr float kStep = 1.0f / 4096.0f;

enum class Operation { Add, FMA, Exp2 };
constexpr std::array kOperations{
    std::pair{Operation::Add, "Add"},
    std::pair{Operation::FMA, "FMA"},
    std::pair{Operation::Exp2, "Exp2"},
};

id<MTLComputePipelineState> makePipeline(id<MTLDevice> device, Operation operation, uint32_t chains) {
  MTLCompileOptions* options = [MTLCompileOptions new];
  options.mathMode = MTLMathModeSafe;
  options.preprocessorMacros = @{
    @"ALU_ADD": @(operation == Operation::Add ? 1 : 0),
    @"ALU_FMA": @(operation == Operation::FMA ? 1 : 0),
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

double timeDispatch(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                    id<MTLBuffer> output, uint32_t threads, uint32_t rounds) {
  @autoreleasepool {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:output offset:0 atIndex:0];
    [encoder setBytes:&rounds length:sizeof(rounds) atIndex:1];
    [encoder setBytes:&kScale length:sizeof(kScale) atIndex:2];
    [encoder setBytes:&kStep length:sizeof(kStep) atIndex:3];
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

void verify(id<MTLBuffer> output, Operation operation, uint32_t threads, uint32_t chains, uint32_t rounds) {
  const auto* values = static_cast<const float*>(output.contents);
  const float tolerance = operation == Operation::Exp2 ? 1e-5f : 0.0f;
  for (uint32_t lane : {0u, threads / 3, threads - 1}) {
    for (uint32_t chain = 0; chain < chains; ++chain) {
      float expected = float((lane & 255u) + chain * 17u) / 256.0f;
      for (uint32_t op = 0; op < rounds * kOpsPerRound; ++op) {
        switch (operation) {
          case Operation::Add: expected += kStep; break;
          case Operation::FMA: expected = std::fma(expected, kScale, kStep); break;
          case Operation::Exp2: expected = std::exp2(-expected); break;
        }
      }
      if (!(std::abs(values[lane * chains + chain] - expected) <= tolerance)) {
        throw std::runtime_error("GPU result disagrees with the FP32 CPU reference");
      }
    }
  }
}

double secondsPerStep(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                      id<MTLBuffer> output, Operation operation, uint32_t threads, uint32_t chains,
                      uint32_t shortRounds, uint32_t longRounds) {
  std::array<double, 31> differences;
  for (size_t i = 0; i < differences.size(); ++i) {
    double shorter, longer;
    if (i % 2 == 0) {
      shorter = timeDispatch(queue, pipeline, output, threads, shortRounds);
      longer = timeDispatch(queue, pipeline, output, threads, longRounds);
    } else {
      longer = timeDispatch(queue, pipeline, output, threads, longRounds);
      shorter = timeDispatch(queue, pipeline, output, threads, shortRounds);
    }
    differences[i] = longer - shorter;
  }
  std::sort(differences.begin(), differences.end());
  const double seconds = differences[differences.size() / 2];
  if (seconds <= 0) throw std::runtime_error("GPU timings could not resolve the added operations");

  timeDispatch(queue, pipeline, output, threads, longRounds);
  verify(output, operation, threads, chains, longRounds);
  return seconds / ((longRounds - shortRounds) * kOpsPerRound);
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

      std::printf("%s | FP32 ALU | normalized to %g MHz\n", device.name.UTF8String, clockHz / 1e6);
      std::printf("%-10s %19s %21s\n", "Operation", "Dependent ns/op", "Ops/device clock");
      for (const auto& [operation, name] : kOperations) {
        id<MTLComputePipelineState> dependent = makePipeline(device, operation, 1);
        id<MTLComputePipelineState> independent = makePipeline(device, operation, kThroughputChains);
        const auto simdWidth = static_cast<uint32_t>(dependent.threadExecutionWidth);
        for (int warmup = 0; warmup < 20; ++warmup) {
          timeDispatch(queue, independent, output, kThroughputThreads, 512);
        }
        const double throughputSeconds = secondsPerStep(
            queue, independent, output, operation, kThroughputThreads, kThroughputChains, 128, 256);
        const double latencySeconds = secondsPerStep(
            queue, dependent, output, operation, simdWidth, 1, 128, 512);
        const double opsPerStep = double(kThroughputThreads) * kThroughputChains;
        std::printf("%-10s %19.3f %21.1f\n", name,
                    latencySeconds * 1e9, opsPerStep / (throughputSeconds * clockHz));
      }
    } catch (const std::exception& error) {
      std::fprintf(stderr, "%s\n", error.what());
      return 1;
    }
  }
}
