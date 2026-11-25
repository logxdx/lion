#include "prompt_cache.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>

#include <cuda_runtime.h>
#include <sys/stat.h>

namespace fs = std::filesystem;

namespace {
constexpr uint64_t MAGIC = 0x31504348444c514cull;  // "LQLDHCP1"
struct FileHeader {
    uint64_t magic, model_key, row_bytes;
    int32_t n, pad;
    uint64_t conv_bytes, swa_bytes, kv_bytes;
};
uint64_t fnv1a(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = (const uint8_t*)p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
void log(const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    fprintf(stderr, "[prompt-cache] %s\n", buf);
}
}  // namespace

PromptCache::PromptCache(le::Engine& e, size_t ram_budget, size_t disk_budget, const std::vector<std::string>& dirs,
                         const std::string& model_path)
    : ram_budget_(ram_budget), disk_budget_(disk_budget), row_bytes_(e.kv_row_bytes()) {
    // entries are only valid for the same model file, KV type and context geometry
    struct stat st{};
    stat(model_path.c_str(), &st);
    const std::string id = fs::absolute(model_path).string() + "|" + std::to_string(st.st_size) + "|" +
                           std::to_string(st.st_mtime) + "|" + std::to_string(row_bytes_) + "|" + (e.kv_q8() ? "q8" : "f16");
    model_key_ = fnv1a(id.data(), id.size());
    if (disk_budget_ > 0) {
        for (const auto& d : dirs) dirs_.push_back(d + "/" + std::to_string(model_key_ % 1000000007ull));
        dir_idx_ = (size_t)-1;
        if (use_next_dir()) writer_ = std::thread([this] { writer_loop(); });
        else {
            log("no usable disk-tier directory: disk tier off");
            disk_budget_ = 0;
        }
    }
    log("RAM budget %zu MiB, disk budget %zu MiB (%s), %zu bytes/token, %zu entries on disk",
        ram_budget_ >> 20, disk_budget_ >> 20, disk_budget_ ? dir_.c_str() : "off", row_bytes_, entries_.size());
}

PromptCache::~PromptCache() {
    {
        std::lock_guard<std::mutex> l(m_);
        stop_ = true;
    }
    wcv_.notify_all();
    if (writer_.joinable()) writer_.join();
    for (auto& en : entries_)
        if (en.kv) cudaFreeHost(en.kv);
}

PromptCache::Entry* PromptCache::find_locked(uint64_t id) {
    for (auto& en : entries_)
        if (en.id == id) return &en;
    return nullptr;
}

bool PromptCache::dir_usable(const std::string& d) {
    std::error_code ec;
    fs::create_directories(d, ec);
    if (ec) return false;
    const std::string probe = d + "/.probe";
    FILE* fp = fopen(probe.c_str(), "wb");
    if (!fp) return false;
    const bool ok = fwrite("ok", 1, 2, fp) == 2 && fflush(fp) == 0;
    fclose(fp);
    fs::remove(probe, ec);
    return ok;
}

bool PromptCache::use_next_dir() {
    for (size_t i = dir_idx_ + 1; i < dirs_.size(); i++) {
        if (!dir_usable(dirs_[i])) {
            log("disk-tier directory unusable: %s", dirs_[i].c_str());
            continue;
        }
        dir_idx_ = i;
        dir_ = dirs_[i];
        try {
            scan_dir();
        } catch (const std::exception& ex) {
            log("scan of %s failed: %s", dir_.c_str(), ex.what());
        }
        if (i > 0) log("disk tier now in %s", dir_.c_str());
        return true;
    }
    return false;
}

void PromptCache::scan_dir() {
    std::error_code ec;
    for (auto& f : fs::directory_iterator(dir_)) {
        if (f.path().extension() != ".lqc") continue;
        FILE* fp = fopen(f.path().c_str(), "rb");
        if (!fp) continue;
        FileHeader h{};
        bool ok = fread(&h, sizeof h, 1, fp) == 1 && h.magic == MAGIC && h.model_key == model_key_ && h.row_bytes == row_bytes_ &&
                  h.n > 0 && h.kv_bytes == (uint64_t)h.n * row_bytes_;
        Entry en;
        if (ok) {
            en.tokens.resize(h.n);
            ok = fread(en.tokens.data(), 4, h.n, fp) == (size_t)h.n;
        }
        fclose(fp);
        if (!ok) { fs::remove(f.path(), ec); continue; }  // stale or corrupt
        en.id = next_id_++;
        en.n = h.n;
        en.kv_bytes = h.kv_bytes;
        en.cp.n_past = h.n;
        en.path = f.path().string();
        en.last_use = 0;
        const auto sz = fs::file_size(f.path(), ec);
        disk_used_ += ec ? 0 : sz;
        entries_.push_back(std::move(en));
    }
}

void PromptCache::store(le::Engine& e, int slot, const std::vector<int>& tokens, const le::Engine::Checkpoint& cp) {
    const int n = (int)tokens.size();
    if (n <= 0 || cp.n_past != n) return;
    const size_t kvb = (size_t)n * row_bytes_;
    if (kvb + cp.bytes() > ram_budget_ && disk_budget_ == 0) return;
    {
        std::lock_guard<std::mutex> l(m_);
        // identical prefix already cached: refresh it
        for (auto& en : entries_)
            if (en.n == n && en.tokens == tokens) { en.last_use = ++tick_; return; }
    }
    uint8_t* kv = nullptr;
    if (cudaMallocHost(&kv, kvb) != cudaSuccess) { cudaGetLastError(); return; }
    e.save_kv(slot, 0, n, kv);
    std::lock_guard<std::mutex> l(m_);
    // older turns of the same conversation (strict prefixes) are superseded by this entry
    for (auto it = entries_.begin(); it != entries_.end();) {
        if (!it->writing && it->n < n && std::equal(it->tokens.begin(), it->tokens.end(), tokens.begin())) {
            if (it->kv) { cudaFreeHost(it->kv); ram_used_ -= it->bytes(); }
            if (!it->path.empty()) { std::error_code ec; disk_used_ -= std::min(disk_used_, (size_t)fs::file_size(it->path, ec)); fs::remove(it->path, ec); }
            it = entries_.erase(it);
        } else ++it;
    }
    Entry en;
    en.id = next_id_++;
    en.tokens = tokens;
    en.n = n;
    en.cp = cp;
    en.kv = kv;
    en.kv_bytes = kvb;
    en.last_use = ++tick_;
    ram_used_ += en.bytes();
    entries_.push_back(std::move(en));
    stores_++;
    enforce_budgets_locked();
}

void PromptCache::enforce_budgets_locked() {
    // RAM over budget: spill least-recently-used entries to disk (or drop them)
    while (ram_used_ > ram_budget_) {
        Entry* lru = nullptr;
        for (auto& en : entries_)
            if (en.kv && !en.writing && (!lru || en.last_use < lru->last_use)) lru = &en;
        if (!lru) break;
        if (disk_budget_ > 0 && lru->path.empty()) {
            lru->writing = true;  // the writer frees the RAM copy once the file is on disk
            wq_.push_back(lru->id);
            wcv_.notify_all();
            spills_++;
            // account as freed now so we do not spill everything at once
            ram_used_ -= lru->bytes();
            lru->last_use = 0;
        } else {
            cudaFreeHost(lru->kv);
            lru->kv = nullptr;
            ram_used_ -= lru->bytes();
            if (lru->path.empty()) {  // not on disk either: gone
                const uint64_t id = lru->id;
                entries_.remove_if([&](const Entry& x) { return x.id == id; });
            }
        }
    }
    // disk over budget: delete least-recently-used files
    while (disk_used_ > disk_budget_) {
        Entry* lru = nullptr;
        for (auto& en : entries_)
            if (!en.path.empty() && !en.kv && !en.writing && (!lru || en.last_use < lru->last_use)) lru = &en;
        if (!lru) break;
        std::error_code ec;
        disk_used_ -= std::min(disk_used_, (size_t)fs::file_size(lru->path, ec));
        fs::remove(lru->path, ec);
        const uint64_t id = lru->id;
        entries_.remove_if([&](const Entry& x) { return x.id == id; });
    }
}

bool PromptCache::write_file(const Entry& en, const std::string& path) {
    const std::string tmp = path + ".tmp";
    FILE* fp = fopen(tmp.c_str(), "wb");
    if (!fp) return false;
    FileHeader h{MAGIC, model_key_, row_bytes_, en.n, 0, en.cp.conv.size(), en.cp.swa.size(), en.kv_bytes};
    bool ok = fwrite(&h, sizeof h, 1, fp) == 1 && fwrite(en.tokens.data(), 4, en.n, fp) == (size_t)en.n &&
              fwrite(en.cp.conv.data(), 1, en.cp.conv.size(), fp) == en.cp.conv.size() &&
              fwrite(en.cp.swa.data(), 1, en.cp.swa.size(), fp) == en.cp.swa.size() &&
              fwrite(en.kv, 1, en.kv_bytes, fp) == en.kv_bytes;
    ok = (fflush(fp) == 0) && ok;
    fclose(fp);
    std::error_code ec;  // no throwing fs calls here: this runs on the writer thread
    if (!ok) { fs::remove(tmp, ec); return false; }
    fs::rename(tmp, path, ec);  // atomic: a crash never leaves a half-written .lqc
    if (ec) { fs::remove(tmp, ec); return false; }
    return true;
}

void PromptCache::writer_loop() {
    for (;;) {
        uint64_t id;
        Entry snapshot;
        {
            std::unique_lock<std::mutex> l(m_);
            wcv_.wait(l, [&] { return stop_ || !wq_.empty(); });
            if (wq_.empty()) return;
            id = wq_.front();
            wq_.pop_front();
            Entry* en = find_locked(id);
            if (!en || !en->kv) continue;
            snapshot.n = en->n;
            snapshot.tokens = en->tokens;
            snapshot.cp = en->cp;
            snapshot.kv = en->kv;  // stays valid: freed only by this thread while writing == true
            snapshot.kv_bytes = en->kv_bytes;
        }
        const std::string name = std::to_string(fnv1a(snapshot.tokens.data(), snapshot.tokens.size() * 4)) + ".lqc";
        std::string path = dir_ + "/" + name;
        bool ok = write_file(snapshot, path);
        if (!ok && !dir_usable(dir_)) {  // the directory died (drive removed, I/O errors): fail over to the next one
            log("disk-tier directory failed: %s", dir_.c_str());
            bool switched;
            {
                std::lock_guard<std::mutex> l(m_);
                // entries stored in the dead directory cannot be read back
                entries_.remove_if([&](const Entry& x) { return !x.path.empty() && !x.writing; });
                disk_used_ = 0;
                switched = use_next_dir();
            }
            if (switched) {
                path = dir_ + "/" + name;
                ok = write_file(snapshot, path);
            } else {
                log("no usable disk-tier directory left: spills are dropped");
            }
        }
        std::lock_guard<std::mutex> l(m_);
        Entry* en = find_locked(id);
        if (!en) continue;
        en->writing = false;
        if (ok) {
            en->path = path;
            std::error_code ec;
            disk_used_ += fs::file_size(path, ec);
        }
        cudaFreeHost(en->kv);  // RAM copy released (its bytes were already un-accounted when the spill was queued)
        en->kv = nullptr;
        if (!ok) entries_.remove_if([&](const Entry& x) { return x.id == id; });
        enforce_budgets_locked();
    }
}

uint64_t PromptCache::lookup(const std::vector<int>& prompt, int min_len, int& n_out) {
    std::lock_guard<std::mutex> l(m_);
    const Entry* best = nullptr;
    for (auto& en : entries_) {
        if (en.n <= min_len || en.n > (int)prompt.size() - 1) continue;
        if (best && en.n <= best->n) continue;
        if (!en.kv && en.path.empty()) continue;
        if (std::equal(en.tokens.begin(), en.tokens.end(), prompt.begin())) best = &en;
    }
    n_out = best ? best->n : 0;
    return best ? best->id : 0;
}

bool PromptCache::read_kv(const Entry& en, std::vector<uint8_t>& kv, le::Engine::Checkpoint& cp) {
    FILE* fp = fopen(en.path.c_str(), "rb");
    if (!fp) return false;
    FileHeader h{};
    bool ok = fread(&h, sizeof h, 1, fp) == 1 && h.magic == MAGIC && h.n == en.n;
    if (ok) ok = fseek(fp, (long)(h.n * 4), SEEK_CUR) == 0;
    cp.n_past = h.n;
    cp.conv.resize(h.conv_bytes);
    cp.swa.resize(h.swa_bytes);
    kv.resize(h.kv_bytes);
    ok = ok && fread(cp.conv.data(), 1, h.conv_bytes, fp) == h.conv_bytes && fread(cp.swa.data(), 1, h.swa_bytes, fp) == h.swa_bytes &&
         fread(kv.data(), 1, h.kv_bytes, fp) == h.kv_bytes;
    fclose(fp);
    return ok;
}

bool PromptCache::restore(le::Engine& e, int slot, uint64_t id, std::string& tier) {
    std::unique_lock<std::mutex> l(m_);
    Entry* en = find_locked(id);
    if (!en) return false;
    en->last_use = ++tick_;
    if (en->kv) {  // RAM hit: keep the lock so the writer cannot spill + free the buffer during the copy
        e.load_kv(slot, 0, en->n, en->kv);
        e.load_checkpoint(slot, en->cp);
        e.write_history(slot, 0, en->tokens.data(), en->n);
        hits_ram_++;
        tier = "ram";
        return true;
    }
    const Entry copy = *en;
    l.unlock();
    std::vector<uint8_t> kv;
    le::Engine::Checkpoint cp;
    if (!read_kv(copy, kv, cp)) {  // unreadable (dead disk, deleted file): forget it, the caller prefills instead
        l.lock();
        std::error_code ec;
        if (Entry* gone = find_locked(id)) {
            disk_used_ -= std::min(disk_used_, (size_t)fs::file_size(gone->path, ec));
            entries_.remove_if([&](const Entry& x) { return x.id == id; });
        }
        return false;
    }
    e.load_kv(slot, 0, copy.n, kv.data());
    e.load_checkpoint(slot, cp);
    e.write_history(slot, 0, copy.tokens.data(), copy.n);
    l.lock();
    hits_disk_++;
    tier = "disk";
    return true;
}

nlohmann::ordered_json PromptCache::stats() {
    std::lock_guard<std::mutex> l(m_);
    size_t in_ram = 0, on_disk = 0;
    for (auto& en : entries_) { in_ram += en.kv != nullptr; on_disk += !en.path.empty(); }
    return {{"entries", entries_.size()}, {"in_ram", in_ram}, {"on_disk", on_disk}, {"ram_mib", ram_used_ >> 20},
            {"disk_mib", disk_used_ >> 20}, {"hits_ram", hits_ram_}, {"hits_disk", hits_disk_}, {"stores", stores_},
            {"spills", spills_}};
}

void PromptCache::flush() {
    if (disk_budget_ == 0) return;
    {
        std::lock_guard<std::mutex> l(m_);
        for (auto& en : entries_)
            if (en.kv && en.path.empty() && !en.writing) {
                en.writing = true;
                ram_used_ -= en.bytes();
                wq_.push_back(en.id);
            }
    }
    wcv_.notify_all();
    for (;;) {  // wait until the writer drained the queue
        {
            std::lock_guard<std::mutex> l(m_);
            bool busy = !wq_.empty();
            for (auto& en : entries_) busy |= en.writing;
            if (!busy) break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    log("flushed RAM tier to %s", dir_.c_str());
}
