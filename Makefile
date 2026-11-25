NVCC  ?= /opt/cuda/bin/nvcc
CUDA  ?= /opt/cuda
ARCH  ?= sm_89
FLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo --default-stream per-thread -Xcompiler -Wall,-Wno-unused-function
BUILD := build

CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wno-unused-function
INC      := -Isrc -Ithird_party -I$(CUDA)/include

HDRS := $(wildcard src/*.h src/*.cuh)
ENGINE := $(BUILD)/model.o $(BUILD)/engine.o

all: $(BUILD)/raon $(BUILD)/raon-server $(BUILD)/raon-tok

$(BUILD)/%.o: src/%.cu $(HDRS) | $(BUILD)
	$(NVCC) $(FLAGS) -Isrc -c $< -o $@

$(BUILD)/raon: $(ENGINE) tools/raon.cu $(HDRS) | $(BUILD)
	$(NVCC) $(FLAGS) -Isrc $(ENGINE) tools/raon.cu -o $@

$(BUILD)/tokenizer.o: src/tokenizer.cpp src/tokenizer.h src/unicode_tables.h src/gguf.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

$(BUILD)/chat.o: src/chat.cpp src/chat.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

$(BUILD)/httplib.o: third_party/httplib/httplib.cpp third_party/httplib/httplib.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -w -Ithird_party/httplib -c $< -o $@

$(BUILD)/raon_server.o: tools/raon_server.cpp src/engine.h src/tokenizer.h src/chat.h src/web_tools.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

$(BUILD)/web_tools.o: src/web_tools.cpp src/web_tools.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

$(BUILD)/raon-server: $(ENGINE) $(BUILD)/tokenizer.o $(BUILD)/chat.o $(BUILD)/httplib.o $(BUILD)/web_tools.o $(BUILD)/raon_server.o | $(BUILD)
	$(NVCC) $(FLAGS) $^ -o $@ -lpthread

$(BUILD)/raon-tok: $(BUILD)/tokenizer.o tools/raon_tok.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INC) $^ -o $@

# checks (see tests/*.py for the reference-based ones)
tests: $(BUILD)/sampling_check $(BUILD)/template_render

$(BUILD)/sampling_check: $(ENGINE) tests/sampling_check.cu | $(BUILD)
	$(NVCC) $(FLAGS) -Isrc $^ -o $@

$(BUILD)/template_render: $(BUILD)/chat.o tests/template_render.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INC) $^ -o $@

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)

# Server tools (web_search, web_fetch, pdf): node dependencies of tools/webtools (raon-server runs node tools.ts)
webtools:
	cd tools/webtools && npm ci --no-audit --no-fund

# Web UI: llama.cpp's prebuilt chat UI (MIT), pinned and checksum-verified, rebranded by brand/rebrand.py.
# raon-server serves build/webui at / (next to the binary; --ui-dir to override).
UI_VERSION ?= b12569
UI_URL     := https://huggingface.co/buckets/ggml-org/llama-ui/resolve/$(UI_VERSION)
UI_TAR     := $(BUILD)/webui-$(UI_VERSION).tar.gz

$(UI_TAR): | $(BUILD)
	curl -fsSL -o $@ $(UI_URL)/dist.tar.gz
	curl -fsSL -o $@.sha256 $(UI_URL)/dist.tar.gz.sha256

webui: $(UI_TAR)
	cd $(BUILD) && echo "$$(cut -d' ' -f1 webui-$(UI_VERSION).tar.gz.sha256)  webui-$(UI_VERSION).tar.gz" | sha256sum -c -
	rm -rf $(BUILD)/webui && mkdir -p $(BUILD)/webui && tar -xzf $(UI_TAR) -C $(BUILD)/webui
	python3 brand/rebrand.py $(BUILD)/webui

.PHONY: all clean tests webui webtools
