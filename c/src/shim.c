/* Opaque llama.cpp handles. Rust sees jf_ctx* and flat arrays. */
#include "shim.h"

#include "ggml-backend.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { JF_BATCH = 1024 };

enum { JF_SEQ_MAX = 16 };

struct jf_ctx {
    struct llama_model *model;
    struct llama_context *ctx;
    const struct llama_vocab *vocab;
    struct mtmd_context *mtmd;
    int32_t n_vocab;
    int32_t n_seq_max;
};

void jf_load_backends(const char *dir) {
    llama_backend_init();
    if (dir && dir[0]) {
        ggml_backend_load_all_from_path(dir);
    } else {
        ggml_backend_load_all();
    }
}

jf_ctx *jf_load(
    const char *path, int32_t n_ctx, int32_t n_threads, int32_t n_gpu_layers, int32_t n_seq_max) {
    struct llama_model_params mp;
    struct llama_model *model;
    struct llama_context_params cp;
    struct llama_context *ctx;
    jf_ctx *h;

    if (!path || n_ctx <= 0 || n_threads < 0 || n_seq_max < 1 || n_seq_max > JF_SEQ_MAX) {
        return NULL;
    }

    mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    model = llama_model_load_from_file(path, mp);
    if (!model) {
        return NULL;
    }

    cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) n_ctx;
    cp.n_batch = JF_BATCH;
    cp.n_ubatch = JF_BATCH;
    cp.n_seq_max = (uint32_t) n_seq_max;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    cp.n_outputs_max = (uint32_t) n_seq_max;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = GGML_TYPE_Q8_0;
    cp.type_v = GGML_TYPE_Q8_0;
    if (n_seq_max > 1) {
        /* Prompts in one batch do not share a suffix. A unified mask is the wrong shape. */
        cp.kv_unified = false;
    }
    ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        llama_model_free(model);
        return NULL;
    }

    h = calloc(1, sizeof(*h));
    if (!h) {
        llama_free(ctx);
        llama_model_free(model);
        return NULL;
    }
    h->model = model;
    h->ctx = ctx;
    h->vocab = llama_model_get_vocab(model);
    h->n_vocab = llama_vocab_n_tokens(h->vocab);
    h->n_seq_max = n_seq_max;
    return h;
}

void jf_free(jf_ctx *h) {
    if (!h) {
        return;
    }
    if (h->mtmd) {
        mtmd_free(h->mtmd);
    }
    if (h->ctx) {
        llama_free(h->ctx);
    }
    if (h->model) {
        llama_model_free(h->model);
    }
    free(h);
}

void jf_backend_free(void) {
    llama_backend_free();
}

int32_t jf_n_vocab(jf_ctx *h) {
    if (!h) {
        return 0;
    }
    return h->n_vocab;
}

int32_t jf_tokenize(jf_ctx *h, const char *text, int32_t add_special, int32_t *out, int32_t cap) {
    size_t len;

    if (!h || !h->vocab || !text || !out || cap < 0) {
        return -1;
    }
    len = strlen(text);
    if (len > (size_t) INT32_MAX) {
        return INT32_MIN;
    }
    return llama_tokenize(h->vocab, text, (int32_t) len, out, cap, add_special != 0, true);
}

int32_t jf_decode(jf_ctx *h, const int32_t *tokens, int32_t n) {
    llama_memory_t mem;
    int32_t off;

    if (!h || !h->ctx || !tokens || n <= 0) {
        return -1;
    }
    mem = llama_get_memory(h->ctx);
    if (!mem) {
        return -1;
    }
    llama_memory_clear(mem, true);

    off = 0;
    while (off < n) {
        struct llama_batch batch;
        int32_t m = n - off;
        int32_t i;
        int32_t rc;

        if (m > JF_BATCH) {
            m = JF_BATCH;
        }
        batch = llama_batch_init(m, 0, 1);
        if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
            llama_batch_free(batch);
            return -1;
        }
        for (i = 0; i < m; i++) {
            batch.token[i] = tokens[off + i];
            batch.pos[i] = off + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (off + i == n - 1) ? 1 : 0;
        }
        batch.n_tokens = m;
        rc = llama_decode(h->ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            return rc;
        }
        off += m;
    }
    return 0;
}

int32_t jf_decode_many(
    jf_ctx *h, const int32_t *const *toks, const int32_t *lens, int32_t n_seq, float *out_logits) {
    llama_memory_t mem;
    int32_t max_n;
    int32_t width;
    int32_t off;
    int32_t s;

    if (!h || !h->ctx || !toks || !lens || !out_logits || n_seq < 1 || n_seq > h->n_seq_max) {
        return -1;
    }
    max_n = 0;
    for (s = 0; s < n_seq; s++) {
        if (!toks[s] || lens[s] <= 0) {
            return -1;
        }
        if (lens[s] > max_n) {
            max_n = lens[s];
        }
    }
    width = JF_BATCH / n_seq;
    if (width < 1) {
        return -1;
    }
    mem = llama_get_memory(h->ctx);
    if (!mem) {
        return -1;
    }
    llama_memory_clear(mem, true);

    off = 0;
    while (off < max_n) {
        struct llama_batch batch;
        int32_t count = 0;
        int32_t t = 0;
        int32_t n_out = 0;
        int32_t out_seq[JF_SEQ_MAX];
        int32_t rc;

        for (s = 0; s < n_seq; s++) {
            int32_t m;
            if (off >= lens[s]) {
                continue;
            }
            m = lens[s] - off;
            if (m > width) {
                m = width;
            }
            count += m;
        }
        batch = llama_batch_init(count, 0, 1);
        if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
            llama_batch_free(batch);
            return -1;
        }
        for (s = 0; s < n_seq; s++) {
            int32_t m;
            int32_t j;
            if (off >= lens[s]) {
                continue;
            }
            m = lens[s] - off;
            if (m > width) {
                m = width;
            }
            for (j = 0; j < m; j++) {
                int32_t pos = off + j;
                batch.token[t] = toks[s][pos];
                batch.pos[t] = pos;
                batch.n_seq_id[t] = 1;
                batch.seq_id[t][0] = s;
                batch.logits[t] = (pos == lens[s] - 1) ? 1 : 0;
                if (batch.logits[t]) {
                    out_seq[n_out] = s;
                    n_out++;
                }
                t++;
            }
        }
        batch.n_tokens = t;
        rc = llama_decode(h->ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            return rc;
        }
        if (n_out > 0) {
            /* Logits of tokens with logits[i] != 0, in the order those tokens were added. */
            const float *all = llama_get_logits(h->ctx);
            if (!all) {
                return -1;
            }
            for (s = 0; s < n_out; s++) {
                const float *row = all + (size_t) s * (size_t) h->n_vocab;
                float *dst = out_logits + (size_t) out_seq[s] * (size_t) h->n_vocab;
                memcpy(dst, row, (size_t) h->n_vocab * sizeof(float));
            }
        }
        off += width;
    }
    return 0;
}

const float *jf_logits(jf_ctx *h) {
    if (!h || !h->ctx) {
        return NULL;
    }
    return llama_get_logits_ith(h->ctx, -1);
}

int32_t jf_vision_load(jf_ctx *h, const char *mmproj, int32_t n_threads) {
    struct mtmd_context_params params;

    if (!h || !h->model || !mmproj) {
        return -1;
    }
    if (h->mtmd) {
        mtmd_free(h->mtmd);
        h->mtmd = NULL;
    }
    params = mtmd_context_params_default();
    params.use_gpu = true;
    params.warmup = false;
    if (n_threads > 0) {
        params.n_threads = n_threads;
    }
    h->mtmd = mtmd_init_from_file(mmproj, h->model, params);
    return h->mtmd ? 0 : -1;
}

const char *jf_media_marker(jf_ctx *h) {
    if (!h || !h->mtmd) {
        return NULL;
    }
    return mtmd_get_marker(h->mtmd);
}

int32_t jf_decode_rgb(
    jf_ctx *h,
    const char *prompt,
    const unsigned char *rgb,
    uint32_t nx,
    uint32_t ny,
    int32_t *n_tokens_out) {
    mtmd_bitmap *bitmap;
    mtmd_input_chunks *chunks;
    mtmd_input_text text;
    const mtmd_bitmap *one;
    llama_memory_t mem;
    llama_pos n_past;
    int32_t rc;
    size_t n_tok;

    if (!h || !h->ctx || !h->mtmd || !prompt || !rgb || nx == 0 || ny == 0) {
        return -1;
    }
    if ((size_t) nx > SIZE_MAX / 3 / (size_t) ny) {
        return -1;
    }
    bitmap = mtmd_bitmap_init(nx, ny, rgb);
    if (!bitmap) {
        return -1;
    }
    chunks = mtmd_input_chunks_init();
    if (!chunks) {
        mtmd_bitmap_free(bitmap);
        return -1;
    }
    text.text = prompt;
    text.text_len = strlen(prompt);
    text.add_special = false;
    text.parse_special = true;
    one = bitmap;
    rc = mtmd_tokenize(h->mtmd, chunks, &text, &one, 1);
    if (rc != 0) {
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bitmap);
        return rc;
    }
    mem = llama_get_memory(h->ctx);
    if (!mem) {
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bitmap);
        return -1;
    }
    llama_memory_clear(mem, true);
    n_past = 0;
    rc = mtmd_helper_eval_chunks(h->mtmd, h->ctx, chunks, 0, 0, JF_BATCH, true, &n_past);
    n_tok = mtmd_helper_get_n_tokens(chunks);
    if (n_tokens_out) {
        *n_tokens_out = n_tok > (size_t) INT32_MAX ? INT32_MAX : (int32_t) n_tok;
    }
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bitmap);
    return rc;
}
