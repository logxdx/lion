// Image decoding (stb_image) and the CPU side of LFM3-VL preprocessing: smart_resize + antialiased bicubic.
// The resampler is a port of llama.cpp's mtmd resize_pillow (itself Pillow's Resample.c) with torchvision's
// int16 weight rounding, so resized pixels match llama.cpp exactly.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "gguf.h"
#include "vision.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

namespace le {

bool decode_image(const uint8_t* data, size_t n, Image& out, std::string& err) {
    int w = 0, h = 0, c = 0;
    unsigned char* px = stbi_load_from_memory(data, (int)n, &w, &h, &c, 3);
    if (!px) {
        err = std::string("cannot decode image: ") + (stbi_failure_reason() ? stbi_failure_reason() : "unknown");
        return false;
    }
    out.w = w;
    out.h = h;
    out.rgb.assign(px, px + (size_t)w * h * 3);
    stbi_image_free(px);
    return true;
}

namespace img {

void smart_resize(int width, int height, int factor, int64_t min_px, int64_t max_px, int& tw, int& th) {
    const double h = height, w = width;
    int h_bar = (int)std::nearbyint(h / factor) * factor;
    int w_bar = (int)std::nearbyint(w / factor) * factor;
    if ((int64_t)h_bar * w_bar > max_px) {
        const double beta = std::sqrt(h * w / (double)max_px);
        h_bar = (int)std::floor(h / beta / factor) * factor;
        w_bar = (int)std::floor(w / beta / factor) * factor;
    } else if ((int64_t)h_bar * w_bar < min_px) {
        const double beta = std::sqrt((double)min_px / (h * w));
        h_bar = (int)std::ceil(h * beta / factor) * factor;
        w_bar = (int)std::ceil(w * beta / factor) * factor;
    }
    tw = std::max(factor, w_bar);
    th = std::max(factor, h_bar);
}

namespace {
constexpr int PRECISION_BITS = 32 - 8 - 2;

double bicubic(double x) {
    constexpr double a = -0.5;
    if (x < 0.0) x = -x;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}

// filter coefficients for one dimension: bounds [out][xmin, xcnt], fixed-point weights [out][ksize]
int precompute(int in_size, int out_size, std::vector<int>& bounds, std::vector<int32_t>& weights) {
    const double scale = (double)in_size / out_size;
    const double filterscale = scale < 1.0 ? 1.0 : scale;
    const double support = 2.0 * filterscale;
    const int ksize = (int)std::ceil(support) * 2 + 1;
    std::vector<double> pre((size_t)out_size * ksize);
    bounds.resize((size_t)out_size * 2);
    for (int xx = 0; xx < out_size; xx++) {
        const double center = (xx + 0.5) * scale;
        const double ss = 1.0 / filterscale;
        double ww = 0.0;
        int xmin = (int)(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = (int)(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;
        int x = 0;
        for (; x < xmax; x++) {
            const double w = bicubic((x + xmin - center + 0.5) * ss);
            pre[(size_t)xx * ksize + x] = w;
            ww += w;
        }
        for (x = 0; x < xmax; x++)
            if (ww != 0.0) pre[(size_t)xx * ksize + x] /= ww;
        for (; x < ksize; x++) pre[(size_t)xx * ksize + x] = 0;
        bounds[xx * 2 + 0] = xmin;
        bounds[xx * 2 + 1] = xmax;
    }
    // torchvision: int16 weights with as many fractional bits as the largest weight allows
    const double wmax = *std::max_element(pre.begin(), pre.end());
    int precision = 0;
    for (; precision < PRECISION_BITS; precision++)
        if ((int)(0.5 + wmax * (1 << (precision + 1))) >= (1 << 15)) break;
    const double fxp = std::ldexp(1.0, precision);
    weights.resize(pre.size());
    for (size_t i = 0; i < pre.size(); i++) {
        const double r = pre[i] * fxp + (pre[i] < 0 ? -0.5 : 0.5);
        weights[i] = (int32_t)r * (1 << (PRECISION_BITS - precision));
    }
    return ksize;
}

inline uint8_t clip8(int v) { return v < 0 ? 0 : v > 255 ? 255 : (uint8_t)v; }
}  // namespace

Image resize_bicubic(const Image& src, int tw, int th) {
    if (tw <= 0 || th <= 0 || tw > 65536 || th > 65536) throw std::runtime_error("resize target out of range");
    if (tw == src.w && th == src.h) return src;
    std::vector<uint8_t> cur = src.rgb;
    int cw = src.w, ch = src.h;
    if (tw != cw) {  // horizontal pass
        std::vector<int> b;
        std::vector<int32_t> k;
        const int ks = precompute(cw, tw, b, k);
        std::vector<uint8_t> out((size_t)tw * ch * 3);
        for (int y = 0; y < ch; y++) {
            const uint8_t* row = cur.data() + (size_t)y * cw * 3;
            uint8_t* dst = out.data() + (size_t)y * tw * 3;
            for (int xx = 0; xx < tw; xx++) {
                const int xmin = b[xx * 2], xcnt = b[xx * 2 + 1];
                const int32_t* kk = &k[(size_t)xx * ks];
                const uint8_t* p = row + (size_t)xmin * 3;
                int32_t s0 = 1 << (PRECISION_BITS - 1), s1 = s0, s2 = s0;
                for (int x = 0; x < xcnt; x++, p += 3) {
                    s0 += p[0] * kk[x];
                    s1 += p[1] * kk[x];
                    s2 += p[2] * kk[x];
                }
                dst[xx * 3 + 0] = clip8(s0 >> PRECISION_BITS);
                dst[xx * 3 + 1] = clip8(s1 >> PRECISION_BITS);
                dst[xx * 3 + 2] = clip8(s2 >> PRECISION_BITS);
            }
        }
        cur.swap(out);
        cw = tw;
    }
    if (th != ch) {  // vertical pass
        std::vector<int> b;
        std::vector<int32_t> k;
        const int ks = precompute(ch, th, b, k);
        const size_t re = (size_t)cw * 3;
        std::vector<uint8_t> out(re * th);
        std::vector<int32_t> acc(re);
        for (int yy = 0; yy < th; yy++) {
            const int ymin = b[yy * 2], ycnt = b[yy * 2 + 1];
            const int32_t* kk = &k[(size_t)yy * ks];
            std::fill(acc.begin(), acc.end(), 1 << (PRECISION_BITS - 1));
            for (int y = 0; y < ycnt; y++) {
                const uint8_t* row = cur.data() + (size_t)(ymin + y) * re;
                const int32_t w = kk[y];
                for (size_t i = 0; i < re; i++) acc[i] += row[i] * w;
            }
            uint8_t* dst = out.data() + (size_t)yy * re;
            for (size_t i = 0; i < re; i++) dst[i] = clip8(acc[i] >> PRECISION_BITS);
        }
        cur.swap(out);
        ch = th;
    }
    Image r;
    r.w = cw;
    r.h = ch;
    r.rgb.swap(cur);
    return r;
}

}  // namespace img

VisionLimits VisionLimits::load(const std::string& path) {
    gguf::File f(path);
    if (!f.has("clip.vision.projector_type") || f.get("clip.vision.projector_type").str != "lfm3vl")
        throw std::runtime_error(path + ": not an lfm3vl mmproj");
    const int64_t patch = (int64_t)f.u("clip.vision.patch_size"), merge = (int64_t)f.u("clip.vision.spatial_merge_size");
    const int64_t area = patch * patch * merge * merge;
    VisionLimits l;
    l.min_px = (int64_t)f.u("clip.vision.image_min_pixels") / area * area;
    l.max_px = std::min<int64_t>(4096, (int64_t)f.u("clip.vision.image_max_pixels") / area) * area;
    return l;
}

}  // namespace le
