NVCC  ?= /opt/cuda/bin/nvcc
ARCH  ?= sm_89
FLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xcompiler -Wall,-Wno-unused-function
BUILD := build

SRC  := src/model.cu src/engine.cu
HDRS := $(wildcard src/*.h src/*.cuh)

all: $(BUILD)/raon

$(BUILD)/%.o: src/%.cu $(HDRS) | $(BUILD)
	$(NVCC) $(FLAGS) -Isrc -c $< -o $@

$(BUILD)/raon: $(BUILD)/model.o $(BUILD)/engine.o tools/raon.cu $(HDRS) | $(BUILD)
	$(NVCC) $(FLAGS) -Isrc $(BUILD)/model.o $(BUILD)/engine.o tools/raon.cu -o $@

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)

.PHONY: all clean
