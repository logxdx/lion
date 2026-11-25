// Default sampling parameters from the GGUF's general.sampling.* keys (the model author's recommendation).
#pragma once
#include <climits>
#include <cmath>

#include "engine.h"
#include "gguf.h"

// Command-line sampling flags; applied on top of the GGUF defaults (unset fields leave them alone).
struct SamplingOverrides {
    float temp = NAN, top_p = NAN, min_p = NAN, penalty = NAN;
    int top_k = INT_MIN, last_n = INT_MIN;
    void apply(le::SamplingParams& sp) const {
        if (!std::isnan(temp)) sp.temperature = temp;
        if (!std::isnan(top_p)) sp.top_p = top_p;
        if (!std::isnan(min_p)) sp.min_p = min_p;
        if (!std::isnan(penalty)) sp.repeat_penalty = penalty;
        if (top_k != INT_MIN) sp.top_k = top_k;
        if (last_n != INT_MIN) sp.repeat_last_n = last_n;
    }
};

// Apply the keys present in the file; parameters the file does not mention keep their SamplingParams defaults
// (top_p 1.0 / min_p 0.0 = off; repetition-penalty window = whole context, as in transformers' generate()).
inline void apply_gguf_sampling(const gguf::File& f, le::SamplingParams& sp) {
    if (f.has("general.sampling.temp")) sp.temperature = (float)f.u("general.sampling.temp");
    if (f.has("general.sampling.top_k")) sp.top_k = (int)f.u("general.sampling.top_k");
    if (f.has("general.sampling.top_p")) sp.top_p = (float)f.u("general.sampling.top_p");
    if (f.has("general.sampling.min_p")) sp.min_p = (float)f.u("general.sampling.min_p");
    if (f.has("general.sampling.penalty_repeat")) sp.repeat_penalty = (float)f.u("general.sampling.penalty_repeat");
    if (f.has("general.sampling.penalty_last_n")) sp.repeat_last_n = (int)f.u("general.sampling.penalty_last_n");
}
