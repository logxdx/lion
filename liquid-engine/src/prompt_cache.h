// Prompt cache tiers below VRAM: evicted sequences (token prefix + end-of-prefix checkpoint + global KV rows)
// live in pinned RAM (LRU, budgeted) and spill to disk files (budgeted, persistent across restarts).
#pragma once
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <list>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "json.hpp"
#include "engine.h"

class PromptCache {
public:
    struct Entry {
        uint64_t id = 0;
        std::vector<int> tokens;      // the cached prefix; n == tokens.size()
        int n = 0;
        le::Engine::Checkpoint cp;    // state after n tokens (conv + SWA window)
        uint8_t* kv = nullptr;        // pinned RAM copy of the global KV rows [0, n) (nullptr: on disk only)
        size_t kv_bytes = 0;
        std::string path;             // disk file ("" if not written yet)
        bool writing = false;         // queued for / being written by the disk thread
        uint64_t last_use = 0;
        size_t bytes() const { return kv_bytes + cp.bytes(); }
    };

    // dirs: disk-tier directories in order of preference. The first one that takes a test write is used; when it dies
    // later (removed drive, I/O errors) spills move on to the next one. All unusable: disk tier off.
    PromptCache(le::Engine& e, size_t ram_budget, size_t disk_budget, const std::vector<std::string>& dirs,
                const std::string& model_path);
    ~PromptCache();

    // Save a slot's state for tokens[0, n) (cp must be the checkpoint at n). Copies KV from the GPU.
    void store(le::Engine& e, int slot, const std::vector<int>& tokens, const le::Engine::Checkpoint& cp);
    // Longest entry that is a prefix of `prompt` with min_len < n <= prompt.size()-1: its id (0 = none) and n.
    uint64_t lookup(const std::vector<int>& prompt, int min_len, int& n_out);
    // Load an entry into a slot (KV rows, checkpoint, token history). tier = "ram" | "disk". false if it vanished.
    bool restore(le::Engine& e, int slot, uint64_t id, std::string& tier);
    nlohmann::ordered_json stats();
    void flush();  // write every RAM-only entry to disk (shutdown)

private:
    size_t ram_budget_, disk_budget_;
    std::string dir_;                 // current disk directory (written by the constructor / the writer thread only)
    std::vector<std::string> dirs_;   // candidates, model-key subdirectory appended
    size_t dir_idx_ = 0;
    uint64_t model_key_ = 0;
    size_t row_bytes_ = 0;
    std::mutex m_;
    std::list<Entry> entries_;
    uint64_t next_id_ = 1, tick_ = 0;
    size_t ram_used_ = 0, disk_used_ = 0;
    // stats
    uint64_t hits_ram_ = 0, hits_disk_ = 0, stores_ = 0, spills_ = 0;
    // disk writer
    std::thread writer_;
    std::condition_variable wcv_;
    std::deque<uint64_t> wq_;
    bool stop_ = false;

    void writer_loop();
    bool use_next_dir();  // advance dir_ to the next candidate that accepts a test write (scans it); false: none left
    static bool dir_usable(const std::string& d);
    bool write_file(const Entry& en, const std::string& path);
    bool read_kv(const Entry& en, std::vector<uint8_t>& kv, le::Engine::Checkpoint& cp);
    void enforce_budgets_locked();
    void scan_dir();
    Entry* find_locked(uint64_t id);
};
