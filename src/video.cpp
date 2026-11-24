// Video decoding for LFM3-VL (see video.h). Ports llama.cpp's mtmd-helper video path for the LFM3 template:
// frame count, frame indices, frame size, the ffmpeg command line (same filter, so the same pixels) and timestamps.
#include "video.h"

#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

extern char** environ;

namespace le {

namespace img {
void smart_resize_video(int width, int height, int n_frames, int& tw, int& th) {
    constexpr int factor = 32, temporal = 2;
    constexpr int64_t min_px = 4096, max_px = 25165824;
    double h = height, w = width;
    if (h < factor || w < factor) {  // the video variant first scales small edges up to the factor
        const double scale = std::max(factor / h, factor / w);
        h = (int)(h * scale);
        w = (int)(w * scale);
    }
    const int64_t t_bar = (int64_t)std::nearbyint((double)n_frames / temporal) * temporal;
    int h_bar = (int)std::nearbyint(h / factor) * factor;
    int w_bar = (int)std::nearbyint(w / factor) * factor;
    if (t_bar * h_bar * w_bar > max_px) {
        const double beta = std::sqrt(n_frames * h * w / max_px);
        h_bar = (int)std::floor(h / beta / factor) * factor;
        w_bar = (int)std::floor(w / beta / factor) * factor;
    } else if (t_bar * h_bar * w_bar < min_px) {
        const double beta = std::sqrt(min_px / (n_frames * h * w));
        h_bar = (int)std::ceil(h * beta / factor) * factor;
        w_bar = (int)std::ceil(w * beta / factor) * factor;
    }
    tw = std::max(factor, w_bar);
    th = std::max(factor, h_bar);
}
}  // namespace img

namespace {
// child process with stdout piped to us and stderr to a temp file (read back for error messages)
struct Proc {
    pid_t pid = -1;
    int out = -1;
    std::string err_path;
    bool start(const std::vector<std::string>& args, std::string& err) {
        int p[2];
        if (pipe2(p, O_CLOEXEC)) { err = strerror(errno); return false; }
        char tmpl[] = "/tmp/liquid-ffmpeg-XXXXXX";
        const int efd = mkstemp(tmpl);
        if (efd < 0) { err = strerror(errno); return false; }
        close(efd);
        err_path = tmpl;
        posix_spawn_file_actions_t fa;
        posix_spawn_file_actions_init(&fa);
        posix_spawn_file_actions_adddup2(&fa, p[1], 1);
        posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
        posix_spawn_file_actions_addopen(&fa, 2, tmpl, O_WRONLY | O_TRUNC, 0600);
        std::vector<char*> argv;
        for (auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
        argv.push_back(nullptr);
        const int rc = posix_spawnp(&pid, argv[0], &fa, nullptr, argv.data(), environ);
        posix_spawn_file_actions_destroy(&fa);
        close(p[1]);
        if (rc) {
            close(p[0]);
            err = "cannot run " + args[0] + ": " + strerror(rc);
            return false;
        }
        out = p[0];
        return true;
    }
    bool read_exact(uint8_t* dst, size_t n) {
        size_t got = 0;
        while (got < n) {
            const ssize_t r = read(out, dst + got, n - got);
            if (r > 0) got += (size_t)r;
            else if (r < 0 && errno == EINTR) continue;
            else return false;
        }
        return true;
    }
    std::string read_all() {
        std::string s;
        char buf[4096];
        ssize_t r;
        while ((r = read(out, buf, sizeof buf)) > 0 || (r < 0 && errno == EINTR))
            if (r > 0) s.append(buf, (size_t)r);
        return s;
    }
    int finish(std::string* err_text = nullptr) {  // close our end, reap, collect stderr
        if (out >= 0) close(out);
        out = -1;
        int st = 0;
        if (pid > 0) {
            while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {}
            pid = -1;
        }
        if (!err_path.empty()) {
            if (err_text) {
                FILE* f = fopen(err_path.c_str(), "r");
                if (f) {
                    char buf[512];
                    size_t n = fread(buf, 1, sizeof buf - 1, f);
                    fclose(f);
                    buf[n] = 0;
                    *err_text = buf;
                }
            }
            unlink(err_path.c_str());
            err_path.clear();
        }
        return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
    }
    ~Proc() {
        if (pid > 0) kill(pid, SIGKILL);
        finish();
    }
};

double parse_rational(const std::string& v) {  // "30000/1001", "2:1", "25"
    size_t sl = v.find('/');
    if (sl == std::string::npos) sl = v.find(':');
    if (sl == std::string::npos) return atof(v.c_str());
    const double a = atof(v.substr(0, sl).c_str()), b = atof(v.substr(sl + 1).c_str());
    return b != 0 ? a / b : 0;
}

std::string tool(const std::string& dir, const char* name) { return dir.empty() ? name : dir + "/" + name; }
}  // namespace

bool decode_video_file(const std::string& path, Video& out, std::string& err, const std::string& ffmpeg_dir) {
    out = Video{};
    // probe
    std::map<std::string, std::string> kv;
    {
        Proc p;
        if (!p.start({tool(ffmpeg_dir, "ffprobe"), "-v", "quiet", "-show_entries",
                      "stream=width,height,sample_aspect_ratio,r_frame_rate,nb_frames,duration,start_time:"
                      "stream_side_data=rotation",
                      "-select_streams", "v:0", "-of",
                      "default=noprint_wrappers=1", path},
                     err))
            return false;
        const std::string s = p.read_all();
        p.finish();
        size_t i = 0;
        while (i < s.size()) {
            size_t nl = s.find('\n', i);
            if (nl == std::string::npos) nl = s.size();
            const std::string line = s.substr(i, nl - i);
            const size_t eq = line.find('=');
            if (eq != std::string::npos) kv[line.substr(0, eq)] = line.substr(eq + 1);
            i = nl + 1;
        }
    }
    out.src_w = atoi(kv["width"].c_str());
    out.src_h = atoi(kv["height"].c_str());
    // display geometry, as ffmpeg delivers the frames (and as llama.cpp sizes them): non-square pixels stretch the
    // width by the sample aspect ratio, and ffmpeg auto-rotates, so a +-90 degree rotation swaps width and height
    const double sar = parse_rational(kv["sample_aspect_ratio"]);
    if (sar > 0 && std::fabs(sar - 1.0) > 1e-6) out.src_w = (int)std::nearbyint(out.src_w * sar);
    const int rot = ((atoi(kv["rotation"].c_str()) % 360) + 360) % 360;
    if (rot == 90 || rot == 270) std::swap(out.src_w, out.src_h);
    out.src_fps = parse_rational(kv["r_frame_rate"]);
    const std::string nb = kv["nb_frames"];
    out.src_frames = (!nb.empty() && nb != "N/A") ? atoi(nb.c_str()) : 0;
    if (out.src_w <= 0 || out.src_h <= 0) { err = "not a video (no video stream found)"; return false; }

    const bool indexed = out.src_frames > 0 && out.src_fps > 0;
    int fw = out.src_w, fh = out.src_h, last = -1;
    std::vector<std::string> args = {tool(ffmpeg_dir, "ffmpeg"), "-nostdin", "-i", path};
    if (indexed) {
        // mtmd_get_video_config: 2 fps, clamp [4, 768] frames, at most the source's
        int n = (int)((double)out.src_frames / out.src_fps * 2.0);
        n = std::clamp(n, 4, 768);
        n = std::min(n, out.src_frames);
        const double step = n > 1 ? (double)(out.src_frames - 1) / (n - 1) : 0.0;
        for (int i = 0; i < n; i++) out.indices.push_back((int)std::nearbyint(i * step));
        img::smart_resize_video(out.src_w, out.src_h, n, fw, fh);
        last = out.indices.back();
        std::string vf = "format=rgb24";
        if (fw != out.src_w || fh != out.src_h)
            vf += ",scale=" + std::to_string(fw) + ":" + std::to_string(fh) +
                  ":flags=bicubic+accurate_rnd+full_chroma_int:param0=0:param1=0.5";
        args.insert(args.end(), {"-vf", vf, "-frames:v", std::to_string(last + 1)});
    } else {
        // no frame count (some webm/mkv): ffmpeg's fps filter at the source size; frames snapped to 32 when encoded
        args.insert(args.end(), {"-vf", "fps=2.000000"});
    }
    args.insert(args.end(), {"-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1", "-loglevel", "error"});

    Proc p;
    if (!p.start(args, err)) return false;
    const size_t fbytes = (size_t)fw * fh * 3;
    std::vector<uint8_t> buf(fbytes);
    size_t want = 0;
    for (int src = 0;; src++) {
        if (indexed && want >= out.indices.size()) break;
        if (!indexed && out.frames.size() >= 768) break;  // same cap as the indexed path
        if (!p.read_exact(buf.data(), fbytes)) break;
        if (indexed && src != out.indices[want]) continue;
        Image im;
        im.w = fw;
        im.h = fh;
        im.rgb = buf;
        out.frames.push_back(std::move(im));
        if (indexed) want++;
    }
    std::string etext;
    const int rc = p.finish(&etext);
    if (out.frames.empty()) {
        err = "ffmpeg decoded no frames" + (etext.empty() ? std::string() : ": " + etext);
        return false;
    }
    (void)rc;
    if (indexed && out.frames.size() < out.indices.size()) {  // fewer frames than nb_frames promised: keep what came
        out.indices.resize(out.frames.size());
    }
    if (!indexed) out.indices.assign(out.frames.size(), -1);
    // one timestamp per pair: mean time of its two frames (a lone last frame: its own time)
    for (size_t first = 0; first < out.frames.size(); first += 2) {
        const size_t second = std::min(first + 1, out.frames.size() - 1);
        const double ts = indexed ? (out.indices[first] / out.src_fps + out.indices[second] / out.src_fps) / 2.0
                                  : (first + 0.5) / 2.0;
        char b[48];
        snprintf(b, sizeof b, "<%.1f seconds>", ts);
        out.stamps.push_back(b);
    }
    return true;
}

bool decode_video(const uint8_t* data, size_t n, Video& out, std::string& err, const std::string& ffmpeg_dir) {
    char tmpl[] = "/tmp/liquid-video-XXXXXX";
    const int fd = mkstemp(tmpl);
    if (fd < 0) { err = strerror(errno); return false; }
    size_t off = 0;
    while (off < n) {
        const ssize_t w = write(fd, data + off, n - off);
        if (w <= 0) { close(fd); unlink(tmpl); err = "cannot write temp file"; return false; }
        off += (size_t)w;
    }
    close(fd);
    const bool ok = decode_video_file(tmpl, out, err, ffmpeg_dir);
    unlink(tmpl);
    return ok;
}

}  // namespace le
