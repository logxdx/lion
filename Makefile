# liquid-engine: LFM3-1.7B inference on an RTX 4050 (sm_89) + Raptor Lake host.
NVCC     ?= /opt/cuda/bin/nvcc
ARCH     ?= sm_89
CXXFLAGS := -O3 -march=native -std=c++17 -Wall
NVFLAGS  := -O3 -std=c++17 -arch=$(ARCH) --use_fast_math -Xcompiler "-O3 -march=native" -lineinfo

BUILD := build
OBJS  := $(BUILD)/engine.o $(BUILD)/tokenizer.o $(BUILD)/main.o

all: $(BUILD)/liquid

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/engine.o: src/engine.cu src/engine.h src/kernels.cuh src/prefill.cuh src/gguf.h | $(BUILD)
	$(NVCC) $(NVFLAGS) -c $< -o $@

$(BUILD)/tokenizer.o: src/tokenizer.cpp src/tokenizer.h src/unicode_data.h src/gguf.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/main.o: src/main.cpp src/engine.h src/tokenizer.h src/gguf.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -I/opt/cuda/include -c $< -o $@

$(BUILD)/liquid: $(OBJS)
	$(NVCC) -arch=$(ARCH) $(OBJS) -o $@ -lcudart

clean:
	rm -rf $(BUILD)

.PHONY: all clean
