// Video input for LFM3-VL, as llama.cpp's mtmd video helper (MTMD_VIDEO_TEMPLATE_LFM3) does it:
//   ffprobe -> n = clamp(trunc(frames / fps * 2), 4, 768) frames at 2 fps, evenly spaced over the source frames
//   -> frame size from smart_resize over the whole clip (factor 32, 4096 .. 25,165,824 px total, pairs of frames)
//   -> ffmpeg scales with swscale bicubic (B=0, C=0.5) -> frames grouped in pairs, each pair one encoder pass.
// In the prompt a video is <|vision_start|> ( "<t seconds>" <|vision_start|> pair <|vision_end|> )* <|vision_end|>,
// t = mean time of the pair's two frames, one decimal.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "vision.h"

namespace le {

struct Video {
    int src_w = 0, src_h = 0, src_frames = 0;
    double src_fps = 0;
    std::vector<Image> frames;           // sampled frames, final size (multiples of 32 except the no-frame-count path)
    std::vector<int> indices;            // source frame index of each sampled frame (-1: fps-filter path)
    std::vector<std::string> stamps;     // per pair of frames: "<t seconds>"
    int n_pairs() const { return (int)(frames.size() + 1) / 2; }
};

// Decode a video file / in-memory bytes (written to a temp file). ffmpeg and ffprobe from PATH (or ffmpeg_dir).
bool decode_video_file(const std::string& path, Video& out, std::string& err, const std::string& ffmpeg_dir = "");
bool decode_video(const uint8_t* data, size_t n, Video& out, std::string& err, const std::string& ffmpeg_dir = "");

namespace img {
// smart_resize, video variant (Qwen3-VL / LFM3-VL video processor): edges up to factor first, area budget over
// n_frames rounded to pairs
void smart_resize_video(int w, int h, int n_frames, int& tw, int& th);
}  // namespace img

}  // namespace le
