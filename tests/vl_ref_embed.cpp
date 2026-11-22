// Reference image embeddings from llama.cpp's libmtmd (for checking src/vision.cu).
// Build: see tests/vl_ref_embed.sh. Usage: vl_ref_embed mmproj.gguf text_model.gguf image out.bin [max_tokens]
// out.bin: int32 n_tokens, int32 n_embd, then n_tokens * n_embd float32. Also prints the chunk layout.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "mtmd.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// VL_DUMP=dir: save intermediate f32 tensors (embedding_norm, layer_out-N, proj_*) as dir/<name>.bin
static const char* g_dump = nullptr;
static bool dump_cb(ggml_tensor* t, bool ask, void*) {
    const char* n = ggml_get_name(t);
    const bool want = t->type == GGML_TYPE_F32 &&
                      (!strncmp(n, "layer_out", 9) || !strcmp(n, "embedding_norm") || !strncmp(n, "proj_", 5) ||
                       !strcmp(n, "patchifier_out") || !strncmp(n, "ffn_inp-", 8));
    if (ask) return want;
    if (want && ggml_is_contiguous(t)) {
        std::vector<float> buf(ggml_nelements(t));
        ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
        std::string path = std::string(g_dump) + "/" + n + ".bin";
        FILE* f = fopen(path.c_str(), "wb");
        int hdr[2] = {(int)t->ne[0], (int)(ggml_nelements(t) / t->ne[0])};
        fwrite(hdr, 4, 2, f);
        fwrite(buf.data(), 4, buf.size(), f);
        fclose(f);
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s mmproj.gguf text_model.gguf image out.bin [max_tokens]\n", argv[0]);
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
    cp.print_timings = true;
    if (argc > 5) cp.image_max_tokens = atoi(argv[5]);
    if ((g_dump = getenv("VL_DUMP"))) { cp.cb_eval = dump_cb; cp.cb_eval_user_data = nullptr; }
    mtmd_context* ctx = mtmd_init_from_file(argv[1], model, cp);
    if (!ctx) return 1;

    int w, h, c;
    unsigned char* px = stbi_load(argv[3], &w, &h, &c, 3);
    if (!px) { fprintf(stderr, "cannot load %s\n", argv[3]); return 1; }
    mtmd_bitmap* bmp = mtmd_bitmap_init(w, h, px);
    stbi_image_free(px);

    const char* prompt = "<__media__>";
    mtmd_input_text txt{prompt, 11, false, true};
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    const mtmd_bitmap* bmps[1] = {bmp};
    if (mtmd_tokenize(ctx, chunks, &txt, bmps, 1) != 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    int n_embd = llama_model_n_embd_inp(model);
    if (n_embd <= 0) n_embd = 2048;  // vocab-only load reports 0; LFM3 width (= the projector output)
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n;
            const llama_token* t = mtmd_input_chunk_get_tokens_text(ch, &n);
            printf("chunk %zu text:", i);
            for (size_t k = 0; k < n; k++) {
                char buf[64];
                int m = llama_token_to_piece(vocab, t[k], buf, sizeof buf, 0, true);
                printf(" %d(%.*s)", t[k], m > 0 ? m : 0, buf);
            }
            printf("\n");
            continue;
        }
        const size_t nt = mtmd_input_chunk_get_n_tokens(ch);
        printf("chunk %zu image: image %dx%d -> %zu tokens, n_pos %d\n", i, w, h, nt, (int)mtmd_input_chunk_get_n_pos(ch));
        if (mtmd_encode_chunk(ctx, ch) != 0) { fprintf(stderr, "encode failed\n"); return 1; }
        {  // warm timing of a second encode
            auto t0 = std::chrono::steady_clock::now();
            mtmd_encode_chunk(ctx, ch);
            printf("encode (warm) %.1f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        const float* e = mtmd_get_output_embd(ctx);
        FILE* f = fopen(argv[4], "wb");
        int hdr[2] = {(int)nt, n_embd};
        fwrite(hdr, 4, 2, f);
        fwrite(e, 4, nt * n_embd, f);
        fclose(f);
        printf("wrote %s: %zu x %d, first %f %f %f\n", argv[4], nt, n_embd, e[0], e[1], e[2]);
    }
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bmp);
    mtmd_free(ctx);
    llama_model_free(model);
    return 0;
}
