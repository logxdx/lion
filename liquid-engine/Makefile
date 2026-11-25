# liquid-engine: LFM3-1.7B inference on an RTX 4050 (sm_89) + Raptor Lake host.
NVCC     ?= /opt/cuda/bin/nvcc
ARCH     ?= sm_89
CUDA_INC ?= /opt/cuda/include
CUDA_LIB ?= /opt/cuda/lib64
CXXFLAGS := -O3 -march=native -std=c++17 -Wall -I$(CUDA_INC)
NVFLAGS  := -O3 -std=c++17 -arch=$(ARCH) --use_fast_math -Xcompiler "-O3 -march=native" -lineinfo

BUILD := build
CORE  := $(BUILD)/engine.o $(BUILD)/tokenizer.o $(BUILD)/vision.o $(BUILD)/image.o $(BUILD)/video.o
SERVER_OBJS := $(BUILD)/server.o $(BUILD)/chat.o $(BUILD)/prompt_cache.o $(BUILD)/web_tools.o $(BUILD)/httplib.o $(BUILD)/decision.o

all: $(BUILD)/liquid $(BUILD)/liquid-server
tests: $(BUILD)/test_engine_api $(BUILD)/test_chat $(BUILD)/test_vision $(BUILD)/test_video $(BUILD)/test_decide

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/engine.o: src/engine.cu src/engine.h src/kernels.cuh src/prefill.cuh src/gemm.cuh src/repack.cuh src/gguf.h | $(BUILD)
	$(NVCC) $(NVFLAGS) -c $< -o $@

$(BUILD)/vision.o: src/vision.cu src/vision.h src/engine.h src/kernels.cuh src/gemm.cuh src/repack.cuh src/gguf.h | $(BUILD)
	$(NVCC) $(NVFLAGS) -c $< -o $@

$(BUILD)/image.o: src/image.cpp src/vision.h src/stb_image.h src/gguf.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/video.o: src/video.cpp src/video.h src/vision.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/tokenizer.o: src/tokenizer.cpp src/tokenizer.h src/unicode_data.h src/gguf.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/main.o: src/main.cpp src/engine.h src/tokenizer.h src/gguf.h src/vision.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/server.o: src/server.cpp src/engine.h src/tokenizer.h src/chat.h src/prompt_cache.h src/web_tools.h src/vision.h src/video.h src/decision.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/decision.o: src/decision.cpp src/decision.h src/tokenizer.h src/gguf.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/chat.o: src/chat.cpp src/chat.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/prompt_cache.o: src/prompt_cache.cpp src/prompt_cache.h src/engine.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/web_tools.o: src/web_tools.cpp src/web_tools.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/httplib.o: src/httplib.cpp src/httplib.h | $(BUILD)
	$(CXX) -O2 -std=c++17 -c $< -o $@

$(BUILD)/liquid: $(CORE) $(BUILD)/main.o
	$(NVCC) -arch=$(ARCH) $^ -o $@ -lcudart

$(BUILD)/liquid-server: $(CORE) $(SERVER_OBJS)
	$(NVCC) -arch=$(ARCH) $^ -o $@ -lcudart -lpthread

$(BUILD)/test_engine_api: tests/test_engine_api.cpp $(CORE)
	$(CXX) $(CXXFLAGS) -c $< -o $(BUILD)/test_engine_api.o
	$(NVCC) -arch=$(ARCH) $(BUILD)/test_engine_api.o $(CORE) -o $@ -lcudart

$(BUILD)/test_vision: tests/test_vision.cpp $(CORE)
	$(CXX) $(CXXFLAGS) -c $< -o $(BUILD)/test_vision.o
	$(NVCC) -arch=$(ARCH) $(BUILD)/test_vision.o $(CORE) -o $@ -lcudart

$(BUILD)/test_decide: tests/test_decide.cpp $(CORE)
	$(CXX) $(CXXFLAGS) -c $< -o $(BUILD)/test_decide.o
	$(NVCC) -arch=$(ARCH) $(BUILD)/test_decide.o $(CORE) -o $@ -lcudart

$(BUILD)/test_video: tests/test_video.cpp $(CORE)
	$(CXX) $(CXXFLAGS) -c $< -o $(BUILD)/test_video.o
	$(NVCC) -arch=$(ARCH) $(BUILD)/test_video.o $(CORE) -o $@ -lcudart

$(BUILD)/test_chat: tests/test_chat.cpp $(BUILD)/chat.o
	$(CXX) $(CXXFLAGS) $^ -o $@

# Built-in tools for liquid-server (web_search, web_fetch: TypeScript; pdf: bash): node deps
webtools:
	cd tools/webtools && npm ci --no-audit --no-fund

# Web UI served by liquid-server at /: llama.cpp's prebuilt bundle (MIT), rebranded as liquine by brand/rebrand.py
UI_VERSION ?= b12569
UI_URL     := https://huggingface.co/buckets/ggml-org/llama-ui/resolve/$(UI_VERSION)
webui: | $(BUILD)
	curl -fsSL -o $(BUILD)/webui-$(UI_VERSION).tar.gz $(UI_URL)/dist.tar.gz
	curl -fsSL -o $(BUILD)/webui-$(UI_VERSION).tar.gz.sha256 $(UI_URL)/dist.tar.gz.sha256
	cd $(BUILD) && echo "$$(cut -d' ' -f1 webui-$(UI_VERSION).tar.gz.sha256)  webui-$(UI_VERSION).tar.gz" | sha256sum -c -
	rm -rf $(BUILD)/webui && mkdir -p $(BUILD)/webui && tar -xzf $(BUILD)/webui-$(UI_VERSION).tar.gz -C $(BUILD)/webui
	python3 brand/rebrand.py $(BUILD)/webui

clean:
	rm -rf $(BUILD)

.PHONY: all tests clean webtools webui
