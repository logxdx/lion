// Minimal mmap-based GGUF v3 reader: metadata key/values + tensor directory.
#pragma once
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace gguf {

enum Type : uint32_t {
    T_U8 = 0, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64
};

// ggml tensor types we care about
enum DType : uint32_t { F32 = 0, F16 = 1, Q8_0 = 8, Q6_K = 14, BF16 = 30 };

struct Value {
    Type type = T_U8;
    Type arr_type = T_U8;
    double num = 0;                    // numeric scalar (all int/float/bool types)
    std::string str;                   // T_STR
    std::vector<double> arr_num;       // numeric arrays
    std::vector<std::string> arr_str;  // string arrays
};

struct Tensor {
    std::string name;
    uint32_t dtype = 0;
    std::vector<uint64_t> ne;  // ne[0] = contiguous dim (ggml order)
    uint64_t offset = 0;       // relative to data section
    const uint8_t* data = nullptr;
    size_t nbytes = 0;
    uint64_t nelem() const { uint64_t n = 1; for (auto d : ne) n *= d; return n; }
};

inline size_t row_bytes(uint32_t dtype, uint64_t n) {
    switch (dtype) {
        case F32: return n * 4;
        case F16: case BF16: return n * 2;
        case Q8_0: return n / 32 * 34;
        case Q6_K: return n / 256 * 210;
        default: throw std::runtime_error("unsupported tensor dtype " + std::to_string(dtype));
    }
}

class File {
public:
    std::map<std::string, Value> kv;
    std::map<std::string, Tensor> tensors;

    explicit File(const std::string& path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("cannot open " + path);
        struct stat st; fstat(fd_, &st); size_ = st.st_size;
        base_ = (const uint8_t*)mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (base_ == MAP_FAILED) throw std::runtime_error("mmap failed");
        madvise((void*)base_, size_, MADV_SEQUENTIAL);
        p_ = base_;
        if (memcmp(p_, "GGUF", 4)) throw std::runtime_error("not a GGUF file");
        p_ += 4;
        uint32_t ver = rd<uint32_t>();
        if (ver < 2) throw std::runtime_error("GGUF version too old");
        uint64_t nt = rd<uint64_t>(), nkv = rd<uint64_t>();
        for (uint64_t i = 0; i < nkv; i++) {
            std::string k = rstr();
            kv[k] = rval((Type)rd<uint32_t>());
        }
        std::vector<Tensor> tl;
        for (uint64_t i = 0; i < nt; i++) {
            Tensor t; t.name = rstr();
            uint32_t nd = rd<uint32_t>();
            for (uint32_t d = 0; d < nd; d++) t.ne.push_back(rd<uint64_t>());
            t.dtype = rd<uint32_t>(); t.offset = rd<uint64_t>();
            tl.push_back(t);
        }
        size_t align = has("general.alignment") ? (size_t)u("general.alignment") : 32;
        size_t off = p_ - base_;
        const uint8_t* data = base_ + (off + align - 1) / align * align;
        for (auto& t : tl) {
            t.data = data + t.offset;
            t.nbytes = row_bytes(t.dtype, t.ne[0]) * (t.nelem() / t.ne[0]);
            if (t.data + t.nbytes > base_ + size_) throw std::runtime_error("tensor out of bounds: " + t.name);
            tensors[t.name] = t;
        }
    }
    ~File() { if (base_) munmap((void*)base_, size_); if (fd_ >= 0) close(fd_); }
    File(const File&) = delete;

    bool has(const std::string& k) const { return kv.count(k) != 0; }
    const Value& get(const std::string& k) const {
        auto it = kv.find(k);
        if (it == kv.end()) throw std::runtime_error("missing GGUF key " + k);
        return it->second;
    }
    double u(const std::string& k) const { return get(k).num; }
    const Tensor& t(const std::string& name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) throw std::runtime_error("missing tensor " + name);
        return it->second;
    }
    const Tensor* tp(const std::string& name) const {
        auto it = tensors.find(name);
        return it == tensors.end() ? nullptr : &it->second;
    }

private:
    int fd_ = -1;
    size_t size_ = 0;
    const uint8_t* base_ = nullptr;
    const uint8_t* p_ = nullptr;

    template <class T> T rd() { T v; memcpy(&v, p_, sizeof(T)); p_ += sizeof(T); return v; }
    std::string rstr() { uint64_t n = rd<uint64_t>(); std::string s((const char*)p_, n); p_ += n; return s; }
    double rnum(Type t) {
        switch (t) {
            case T_U8: return rd<uint8_t>();   case T_I8: return rd<int8_t>();
            case T_U16: return rd<uint16_t>(); case T_I16: return rd<int16_t>();
            case T_U32: return rd<uint32_t>(); case T_I32: return rd<int32_t>();
            case T_F32: return rd<float>();    case T_BOOL: return rd<uint8_t>();
            case T_U64: return (double)rd<uint64_t>(); case T_I64: return (double)rd<int64_t>();
            case T_F64: return rd<double>();
            default: throw std::runtime_error("bad GGUF scalar type");
        }
    }
    Value rval(Type t) {
        Value v; v.type = t;
        if (t == T_STR) { v.str = rstr(); return v; }
        if (t == T_ARR) {
            v.arr_type = (Type)rd<uint32_t>();
            uint64_t n = rd<uint64_t>();
            if (v.arr_type == T_STR) { v.arr_str.reserve(n); for (uint64_t i = 0; i < n; i++) v.arr_str.push_back(rstr()); }
            else { v.arr_num.reserve(n); for (uint64_t i = 0; i < n; i++) v.arr_num.push_back(rnum(v.arr_type)); }
            return v;
        }
        v.num = rnum(t);
        return v;
    }
};

}  // namespace gguf
