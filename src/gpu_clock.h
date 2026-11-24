#pragma once

#include <cmath>
#include <cstdlib>
#include <stdexcept>

inline double gpuClockHz(int argc, char* argv[]) {
  if (argc != 2) throw std::runtime_error("Expected one argument: GPU clock in MHz");
  char* end = nullptr;
  const double hz = std::strtod(argv[1], &end) * 1e6;
  if (end == argv[1] || *end != '\0' || !std::isfinite(hz) || hz <= 0) {
    throw std::runtime_error("GPU clock must be a positive frequency in MHz");
  }
  return hz;
}
