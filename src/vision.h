// LFM3-VL vision encoder (mmproj, projector type "lfm3vl") on the GPU.
//
// image (JPEG/PNG/...) -> smart_resize to multiples of 32 (64..4096 tokens) + torchvision-style antialiased bicubic
// -> 16x16 patches -> 14 blocks (8 x 3x3 depthwise-conv mixer, 6 x GQA attention with 2D RoPE; RMSNorm, SwiGLU)
// -> LayerNorm, 2x2 merge, Linear-GELU-Linear -> one 2048-dim embedding per 32x32 pixel block.
// The embeddings replace the <|image|> token between <|vision_start|> and <|vision_end|> in the LLM prompt.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace le {

struct Image {
    int w = 0, h = 0;
    std::vector<uint8_t> rgb;  // w * h * 3
};
// Decode JPEG / PNG / BMP / GIF (first frame) / PSD / TGA / HDR / PNM bytes. false + err on failure.
bool decode_image(const uint8_t* data, size_t n, Image& out, std::string& err);

namespace img {
// Qwen2-VL smart_resize (as llama.cpp / transformers): edges rounded to multiples of factor, area kept in
// [min_px, max_px]
void smart_resize(int w, int h, int factor, int64_t min_px, int64_t max_px, int& tw, int& th);
// Separable antialiased bicubic (a = -0.5) with torchvision's int16 weights, as llama.cpp's
// RESIZE_ALGO_BICUBIC | RESIZE_ALGO_FLAG_TORCH (bit-exact with it)
Image resize_bicubic(const Image& src, int tw, int th);
}  // namespace img

// Image token budget of an lfm3vl mmproj, from its metadata only (no GPU): the prompt layout is known before
// the encoder runs. As llama.cpp: min from the GGUF, max capped at 4096 tokens.
struct VisionLimits {
    int64_t min_px = 64 * 1024, max_px = 4096 * 1024;
    static VisionLimits load(const std::string& mmproj_path);
    void target_size(int w, int h, int& tw, int& th) const { img::smart_resize(w, h, 32, min_px, max_px, tw, th); }
    int n_tokens(int w, int h) const {
        int tw, th;
        target_size(w, h, tw, th);
        return (tw / 32) * (th / 32);
    }
};

class Vision {
public:
    explicit Vision(const std::string& mmproj_path, bool verbose = true);
    ~Vision();
    Vision(const Vision&) = delete;
    Vision& operator=(const Vision&) = delete;

    // size the image is resized to (multiples of 32) and the number of embeddings it produces
    void target_size(int w, int h, int& tw, int& th) const;
    int n_tokens(int w, int h) const;
    int n_embd() const { return 2048; }

    // Encode one image: returns a device pointer to [n_tok][2048] f32, valid until the next encode().
    const float* encode(const Image& img, int& n_tok);
    std::vector<float> encode_host(const Image& img, int& n_tok);

    double last_ms() const { return last_ms_; }
    size_t gpu_bytes() const { return gpu_bytes_; }

    struct Impl;
private:
    std::unique_ptr<Impl> p_;
    double last_ms_ = 0;
    size_t gpu_bytes_ = 0;
};

}  // namespace le
