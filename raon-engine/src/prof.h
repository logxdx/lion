// RAON_PROFILE=1: synchronising CUDA events between kernel groups; per-group time per forward call at exit.
#pragma once
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace raon {

struct Prof {
    bool on = getenv("RAON_PROFILE") != nullptr;
    cudaEvent_t last{}, cur{};
    std::map<std::string, double> ms;
    int calls = 0;
    Prof() {
        if (on) { cudaEventCreate(&last); cudaEventCreate(&cur); }
    }
    void start() {
        if (on) cudaEventRecord(last);
    }
    void mark(const char* name) {
        if (!on) return;
        cudaEventRecord(cur);
        const cudaError_t e = cudaEventSynchronize(cur);
        if (e != cudaSuccess) {
            fprintf(stderr, "profile: CUDA error %s surfaced after group '%s'\n", cudaGetErrorString(e), name);
            abort();
        }
        float t = 0;
        cudaEventElapsedTime(&t, last, cur);
        ms[name] += t;
        std::swap(last, cur);
    }
    void report() {
        if (!on || !calls) return;
        std::vector<std::pair<double, std::string>> v;
        double tot = 0;
        for (auto& [k, x] : ms) { v.push_back({x, k}); tot += x; }
        std::sort(v.rbegin(), v.rend());
        fprintf(stderr, "profile (%d forward chunks, per chunk):\n", calls);
        for (auto& [x, k] : v) fprintf(stderr, "  %-18s %8.3f ms  %5.1f%%\n", k.c_str(), x / calls, 100 * x / tot);
        fprintf(stderr, "  %-18s %8.3f ms\n", "total", tot / calls);
    }
};

}  // namespace raon
