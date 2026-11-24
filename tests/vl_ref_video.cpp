// Reference embeddings for video frame pairs from llama.cpp's libmtmd (checks src/vision.cu's video path).
// Frames are given as image files in order (extract them with the same ffmpeg command the engine uses); consecutive
// frames are marked as mergeable video bitmaps, so libmtmd groups them in pairs (temporal_patch_size 2) as its
// video helper does.
// usage: vl_ref_video mmproj.gguf text_model.gguf out.bin frame0 frame1 [frame2 frame3 ...]
// out.bin: int32 n_chunks, then per media chunk: int32 n_tokens, int32 n_embd, n_tokens * n_embd float32
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "llama.h"
#include "mtmd.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s mmproj.gguf text_model.gguf out.bin frame0 frame1 [...]\n", argv[0]);
        return 1;
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model* model = llama_model_load_from_file(argv[2], mp);
    if (!model) return 1;
    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = getenv("VL_CPU") == nullptr;
    cp.warmup = false;
    mtmd_context* ctx = mtmd_init_from_file(argv[1], model, cp);
    if (!ctx) return 1;

    std::vector<mtmd_bitmap*> bmps;
    std::string prompt;
    for (int i = 4; i < argc; i++) {
        int w, h, c;
        unsigned char* px = stbi_load(argv[i], &w, &h, &c, 3);
        if (!px) { fprintf(stderr, "cannot load %s\n", argv[i]); return 1; }
        mtmd_bitmap* b = mtmd_bitmap_init(w, h, px);
        stbi_image_free(px);
        mtmd_bitmap_set_is_video(b, true);
        mtmd_bitmap_set_mergeable(b, true);
        bmps.push_back(b);
        prompt += mtmd_default_marker();
    }
    mtmd_input_text txt{prompt.c_str(), prompt.size(), false, true};
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    if (mtmd_tokenize(ctx, chunks, &txt, (const mtmd_bitmap**)bmps.data(), bmps.size()) != 0) {
        fprintf(stderr, "tokenize failed\n");
        return 1;
    }
    const int n_embd = 2048;
    FILE* f = fopen(argv[3], "wb");
    int n_media = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++)
        n_media += mtmd_input_chunk_get_type(mtmd_input_chunks_get(chunks, i)) != MTMD_INPUT_CHUNK_TYPE_TEXT;
    fwrite(&n_media, 4, 1, f);
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n;
            mtmd_input_chunk_get_tokens_text(ch, &n);
            printf("chunk %zu: text, %zu tokens\n", i, n);
            continue;
        }
        const size_t nt = mtmd_input_chunk_get_n_tokens(ch);
        if (mtmd_encode_chunk(ctx, ch) != 0) { fprintf(stderr, "encode failed\n"); return 1; }
        const float* e = mtmd_get_output_embd(ctx);
        int hdr[2] = {(int)nt, n_embd};
        fwrite(hdr, 4, 2, f);
        fwrite(e, 4, nt * n_embd, f);
        printf("chunk %zu: media, %zu tokens, first %f %f\n", i, nt, e[0], e[1]);
    }
    fclose(f);
    for (auto* b : bmps) mtmd_bitmap_free(b);
    mtmd_input_chunks_free(chunks);
    mtmd_free(ctx);
    llama_model_free(model);
    return 0;
}
