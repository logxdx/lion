#!/bin/sh
# Build tests/vl_ref_embed against a llama.cpp checkout (headers) and build (libs).
#   LLAMA_SRC=/path/to/llama.cpp LLAMA_LIB=~/llama.cpp/current tests/vl_ref_embed.sh
set -e
S=${LLAMA_SRC:?set LLAMA_SRC to a llama.cpp source tree}
L=${LLAMA_LIB:?set LLAMA_LIB to the directory with libmtmd.so / libllama.so}
mkdir -p build
g++ -O2 -std=c++17 -o build/vl_ref_embed tests/vl_ref_embed.cpp \
    -I"$S/include" -I"$S/ggml/include" -I"$S/tools/mtmd" -I"$S/vendor/stb" \
    -L"$L" -lmtmd -lllama -lggml -lggml-base -Wl,-rpath,"$L"
echo built build/vl_ref_embed
