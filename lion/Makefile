# lion: the gateway (lion.py, stdlib Python: nothing to build) and its web UI (llama.cpp's, branded Lion)
BUILD := build

# Web UI: llama.cpp's prebuilt chat UI (MIT), pinned and checksum-verified, rebranded by brand/rebrand.py
UI_VERSION ?= b12569
UI_URL     := https://huggingface.co/buckets/ggml-org/llama-ui/resolve/$(UI_VERSION)
UI_TAR     := $(BUILD)/webui-$(UI_VERSION).tar.gz

all: webui

$(BUILD):
	mkdir -p $(BUILD)

$(UI_TAR): | $(BUILD)
	curl -fsSL -o $@ $(UI_URL)/dist.tar.gz
	curl -fsSL -o $@.sha256 $(UI_URL)/dist.tar.gz.sha256

webui: $(UI_TAR)
	cd $(BUILD) && echo "$$(cut -d' ' -f1 webui-$(UI_VERSION).tar.gz.sha256)  webui-$(UI_VERSION).tar.gz" | sha256sum -c -
	rm -rf $(BUILD)/webui && mkdir -p $(BUILD)/webui && tar -xzf $(UI_TAR) -C $(BUILD)/webui
	python3 brand/rebrand.py $(BUILD)/webui

# brand/webui/ (icons, splash screens) from brand/lion-mark.svg; needs rsvg-convert + ImageMagick
assets: $(UI_TAR)
	rm -rf $(BUILD)/webui-src && mkdir -p $(BUILD)/webui-src && tar -xzf $(UI_TAR) -C $(BUILD)/webui-src
	python3 brand/build_assets.py $(BUILD)/webui-src

# node dependencies of the server tools the gateway runs (shared with liquid-engine)
webtools:
	$(MAKE) -C ../liquid-engine webtools

clean:
	rm -rf $(BUILD)

.PHONY: all webui assets webtools clean
