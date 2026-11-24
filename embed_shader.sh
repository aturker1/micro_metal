#!/bin/sh
set -eu

printf 'inline constexpr char %s[] = R"METAL(\n' "${2:-matmul2dShader}"
cat "$1"
printf '\n%s\n' ')METAL";'
