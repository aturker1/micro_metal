#!/bin/sh
set -eu

printf '%s\n' 'inline constexpr char matmul2dShader[] = R"MATMUL2D('
cat "$1"
printf '\n%s\n' ')MATMUL2D";'
