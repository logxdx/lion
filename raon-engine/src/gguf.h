// Minimal read-only GGUF v3 reader over an mmap'd file.
#pragma once
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace raon {

enum GType : uint32_t { G_F32 = 0, G_F16 = 1, G_Q8_0 = 8, G_Q6_K = 14, G_BF16 = 30 };

inline size_t gtype_bytes(uint32_t t, int64_t n) {
    switch (t) {
        case G_F32: return 4 * n;
        case G_F16: case G_BF16: return 2 * n;
        case G_Q8_0: return n / 32 * 34;
        case G_Q6_K: return n / 256 * 210;
    }
    throw std::runtime_error("unsupported GGUF tensor type " + std::to_string(t));
}

struct GTensor {
    std::string name;
    std::vector<int64_t> ne;  // ne[0] is the contiguous dimension
    uint32_t type = 0;
    const uint8_t* data = nullptr;
    size_t nbytes = 0;
    int64_t nelem() const { int64_t n = 1; for (auto v : ne) n *= v; return n; }
};

struct GValue {
    uint32_t type = 0;
    uint64_t u = 0;
    int64_t i = 0;
    double f = 0;
    std::string s;
    uint32_t arr_type = 0;
    std::vector<GValue> arr;
};

class GGUF {
public:
    explicit GGUF(const std::string& path) {
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("cannot open " + path);
        struct stat st;
        fstat(fd_, &st);
        size_ = st.st_size;
        base_ = (const uint8_t*)mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (base_ == MAP_FAILED) throw std::runtime_error("mmap failed: " + path);
        p_ = base_;
        if (memcmp(p_, "GGUF", 4)) throw std::runtime_error("not a GGUF file: " + path);
        p_ += 4;
        const uint32_t ver = rd<uint32_t>();
        if (ver < 2) throw std::runtime_error("GGUF version too old");
        const uint64_t nt = rd<uint64_t>(), nkv = rd<uint64_t>();
        for (uint64_t k = 0; k < nkv; k++) {
            std::string key = rds();
            kv_[key] = rdv(rd<uint32_t>());
        }
        std::vector<std::pair<GTensor, uint64_t>> infos;
        for (uint64_t t = 0; t < nt; t++) {
            GTensor g;
            g.name = rds();
            const uint32_t nd = rd<uint32_t>();
            for (uint32_t d = 0; d < nd; d++) g.ne.push_back((int64_t)rd<uint64_t>());
            g.type = rd<uint32_t>();
            const uint64_t off = rd<uint64_t>();
            infos.push_back({g, off});
        }
        uint64_t align = 32;
        if (has("general.alignment")) align = u("general.alignment");
        size_t pos = p_ - base_;
        pos = (pos + align - 1) / align * align;
        for (auto& [g, off] : infos) {
            g.data = base_ + pos + off;
            g.nbytes = gtype_bytes(g.type, g.nelem());
            if (g.data + g.nbytes > base_ + size_) throw std::runtime_error("tensor out of file bounds: " + g.name);
            tensors_[g.name] = g;
        }
    }
    ~GGUF() {
        if (base_ && base_ != MAP_FAILED) munmap((void*)base_, size_);
        if (fd_ >= 0) close(fd_);
    }
    GGUF(const GGUF&) = delete;

    bool has(const std::string& k) const { return kv_.count(k); }
    const GValue& get(const std::string& k) const {
        auto it = kv_.find(k);
        if (it == kv_.end()) throw std::runtime_error("missing GGUF key " + k);
        return it->second;
    }
    uint64_t u(const std::string& k) const { const auto& v = get(k); return v.type == 6 || v.type == 12 ? (uint64_t)v.f : v.u; }
    double f(const std::string& k) const { const auto& v = get(k); return (v.type == 6 || v.type == 12) ? v.f : (double)v.u; }
    const std::string& s(const std::string& k) const { return get(k).s; }

    bool has_tensor(const std::string& n) const { return tensors_.count(n); }
    const GTensor& t(const std::string& n) const {
        auto it = tensors_.find(n);
        if (it == tensors_.end()) throw std::runtime_error("missing tensor " + n);
        return it->second;
    }

private:
    template <class T>
    T rd() { T v; memcpy(&v, p_, sizeof(T)); p_ += sizeof(T); return v; }
    std::string rds() { const uint64_t n = rd<uint64_t>(); std::string s((const char*)p_, n); p_ += n; return s; }
    GValue rdv(uint32_t type) {
        GValue v;
        v.type = type;
        switch (type) {
            case 0: v.u = rd<uint8_t>(); break;
            case 1: v.i = rd<int8_t>(); v.u = (uint64_t)v.i; break;
            case 2: v.u = rd<uint16_t>(); break;
            case 3: v.i = rd<int16_t>(); v.u = (uint64_t)v.i; break;
            case 4: v.u = rd<uint32_t>(); break;
            case 5: v.i = rd<int32_t>(); v.u = (uint64_t)v.i; break;
            case 6: v.f = rd<float>(); break;
            case 7: v.u = rd<uint8_t>(); break;
            case 8: v.s = rds(); break;
            case 9: {
                v.arr_type = rd<uint32_t>();
                const uint64_t n = rd<uint64_t>();
                v.arr.reserve(n);
                for (uint64_t i = 0; i < n; i++) v.arr.push_back(rdv(v.arr_type));
                break;
            }
            case 10: v.u = rd<uint64_t>(); break;
            case 11: v.i = rd<int64_t>(); v.u = (uint64_t)v.i; break;
            case 12: v.f = rd<double>(); break;
            default: throw std::runtime_error("bad GGUF value type " + std::to_string(type));
        }
        return v;
    }

    int fd_ = -1;
    size_t size_ = 0;
    const uint8_t* base_ = nullptr;
    const uint8_t* p_ = nullptr;
    std::unordered_map<std::string, GValue> kv_;
    std::unordered_map<std::string, GTensor> tensors_;
};

}  // namespace raon
