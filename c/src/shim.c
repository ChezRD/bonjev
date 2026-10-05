/* Opaque llama.cpp handles. Rust sees jf_ctx* and flat arrays. */
#include "shim.h"

#include "ggml-backend.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { JF_BATCH = 1024 };

/* BONJEV_TIMING=1 prints per-request stage timings to stderr. */
static int timing_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *value = getenv("BONJEV_TIMING");
        cached = (value && value[0] && value[0] != '0') ? 1 : 0;
    }
    return cached;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1000.0 + (double) ts.tv_nsec / 1e6;
}

enum { JF_BATCH_MIN = 32, JF_BATCH_MAX = 8192 };

/* One `<path>[,scale]` adapter entry. The scale defaults to 1. */
static struct llama_adapter_lora *jf_lora_init(struct llama_model *model, const char *raw, float *scale_out) {
    char path[4096];
    const char *comma = strrchr(raw, ',');
    size_t len = comma ? (size_t) (comma - raw) : strlen(raw);
    float scale = 1.0f;
    if (len == 0 || len >= sizeof(path)) {
        return NULL;
    }
    memcpy(path, raw, len);
    path[len] = '\0';
    if (comma) {
        char *end = NULL;
        float value = strtof(comma + 1, &end);
        if (end && end != comma + 1) {
            scale = value;
        }
    }
    *scale_out = scale;
    return llama_adapter_lora_init(model, path);
}

enum { JF_LORA_MAX = 8 };
enum { JF_SEQ_MAX = 16 };

/* One saved prefix per pattern family. The model is hybrid (attention plus
   recurrent layers), so a KV tail cannot be trimmed: a matching prefix is
   restored from a checkpoint instead. Styles that alternate between requests
   (per-kind routing) keep one slot each. */
enum { JF_CKPT_MAX = 4 };

struct jf_ckpt {
    int32_t n;
    int32_t *tok;
    uint8_t *data;
    size_t size;
    uint64_t used_at;
};

struct jf_ctx {
    struct llama_model *model;
    struct llama_context *ctx;
    struct llama_adapter_lora *loras[JF_LORA_MAX];
    float lora_scales[JF_LORA_MAX];
    int32_t n_lora;
    const struct llama_vocab *vocab;
    struct mtmd_context *mtmd;
    int32_t n_vocab;
    int32_t n_seq_max;
    int32_t n_ctx_seq;
    /* n_batch / n_ubatch for prefill chunking. `BONJEV_BATCH` overrides. */
    int32_t batch;
    int32_t cache_len[JF_SEQ_MAX];
    /* Token ids already sitting in the KV cache, one row per sequence. */
    int32_t *cache_tok;
    struct jf_ckpt ckpt[JF_CKPT_MAX];
    uint64_t ckpt_clock;
    /* Tokens reused from KV vs prompt length, per sequence, for the last decode call. */
    int32_t last_nseq;
    int32_t last_keep[JF_SEQ_MAX];
    int32_t last_total[JF_SEQ_MAX];
};

void jf_load_backends(const char *dir) {
    llama_backend_init();
    if (dir && dir[0]) {
        ggml_backend_load_all_from_path(dir);
    } else {
        ggml_backend_load_all();
    }
}

int32_t jf_load(
    const char *path,
    const char *lora,
    int32_t n_ctx,
    int32_t n_threads,
    int32_t n_gpu_layers,
    jf_ctx **out) {
    struct llama_model_params mp;
    struct llama_model *model;
    struct llama_context_params cp;
    struct llama_context *ctx;
    struct llama_adapter_lora *loras[JF_LORA_MAX];
    float lora_scales[JF_LORA_MAX];
    int32_t n_lora;
    int32_t li;
    jf_ctx *h;
    int32_t n_seq_max;
    int32_t batch;

    if (!out) {
        return JF_LOAD_ARGS;
    }
    *out = NULL;
    if (!path || n_ctx <= 0 || n_threads < 0) {
        return JF_LOAD_ARGS;
    }

    batch = JF_BATCH;
    {
        const char *raw = getenv("BONJEV_BATCH");
        if (raw && raw[0]) {
            int value = atoi(raw);
            if (value >= JF_BATCH_MIN && value <= JF_BATCH_MAX) {
                batch = (int32_t) value;
            }
        }
    }

    mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    model = llama_model_load_from_file(path, mp);
    if (!model) {
        return JF_LOAD_MODEL;
    }

    cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) n_ctx;
    /* Three sequences, one unified KV of size n_ctx. A single style still
       uses one sequence and may fill the whole window. `prompt_style=top3`
       decodes three leads for the same question in one batch; those three
       prompts share the cell pool, so together they must fit in n_ctx. */
    cp.n_seq_max = 3;
    cp.n_outputs_max = 3;
    cp.kv_unified = true;
    cp.n_batch = (uint32_t) batch;
    cp.n_ubatch = (uint32_t) batch;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = GGML_TYPE_Q4_0;
    cp.type_v = GGML_TYPE_Q4_0;
    {
        /* BONJEV_KV_TYPE=f16 (or q8_0) switches the K/V cache from Q4_0. */
        const char *raw = getenv("BONJEV_KV_TYPE");
        if (raw && (strcmp(raw, "f16") == 0 || strcmp(raw, "fp16") == 0)) {
            cp.type_k = GGML_TYPE_F16;
            cp.type_v = GGML_TYPE_F16;
        } else if (raw && strcmp(raw, "q8_0") == 0) {
            cp.type_k = GGML_TYPE_Q8_0;
            cp.type_v = GGML_TYPE_Q8_0;
        }
    }
    ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        llama_model_free(model);
        return JF_LOAD_CONTEXT;
    }

    /* `lora` = <path>[,scale][;<path>[,scale]...] attaches N adapters. */
    n_lora = 0;
    {
        const char *raw = lora;
        if (raw && raw[0]) {
            const char *p = raw;
            while (*p && n_lora < JF_LORA_MAX) {
                const char *semi = strchr(p, ';');
                size_t len = semi ? (size_t) (semi - p) : strlen(p);
                char entry[4096];
                if (len == 0 || len >= sizeof(entry)) {
                    break;
                }
                memcpy(entry, p, len);
                entry[len] = '\0';
                float sc = 1.0f;
                struct llama_adapter_lora *a = jf_lora_init(model, entry, &sc);
                if (!a) {
                    fprintf(stderr, "[bonjev] failed to load lora: %s\n", entry);
                    for (li = 0; li < n_lora; li++) {
                        llama_adapter_lora_free(loras[li]);
                    }
                    llama_free(ctx);
                    llama_model_free(model);
                    return JF_LOAD_LORA;
                }
                loras[n_lora] = a;
                lora_scales[n_lora] = sc;
                fprintf(stderr, "[bonjev] lora[%d]=%s scale=%g\n", n_lora, entry, sc);
                n_lora++;
                p = semi ? semi + 1 : p + len;
            }
            if (n_lora == 0 ||
                llama_set_adapters_lora(ctx, loras, n_lora, lora_scales) != 0) {
                fprintf(stderr, "[bonjev] failed to attach loras: %s\n", raw);
                for (li = 0; li < n_lora; li++) {
                    llama_adapter_lora_free(loras[li]);
                }
                llama_free(ctx);
                llama_model_free(model);
                return JF_LOAD_LORA;
            }
        }
    }

    h = calloc(1, sizeof(*h));
    if (!h) {
        llama_free(ctx);
        llama_model_free(model);
        return JF_LOAD_INTERNAL;
    }
    h->model = model;
    h->ctx = ctx;
    h->n_lora = n_lora;
    for (li = 0; li < n_lora; li++) {
        h->loras[li] = loras[li];
        h->lora_scales[li] = lora_scales[li];
    }
    h->vocab = llama_model_get_vocab(model);
    h->n_vocab = llama_vocab_n_tokens(h->vocab);
    h->batch = batch;
    n_seq_max = (int32_t) llama_n_seq_max(ctx);
    if (n_seq_max < 1 || n_seq_max > JF_SEQ_MAX) {
        free(h);
        llama_free(ctx);
        llama_model_free(model);
        return JF_LOAD_INTERNAL;
    }
    h->n_seq_max = n_seq_max;
    h->n_ctx_seq = (int32_t) llama_n_ctx_seq(ctx);
    if (h->n_ctx_seq <= 0) {
        free(h);
        llama_free(ctx);
        llama_model_free(model);
        return JF_LOAD_INTERNAL;
    }
    h->cache_tok = calloc((size_t) n_seq_max * (size_t) h->n_ctx_seq, sizeof(int32_t));
    if (!h->cache_tok) {
        free(h->cache_tok);
        free(h);
        llama_free(ctx);
        llama_model_free(model);
        return JF_LOAD_INTERNAL;
    }
    fprintf(
        stderr,
        "[bonjev] ctx=%u ctx_seq=%u seq_max=%u\n",
        llama_n_ctx(ctx),
        llama_n_ctx_seq(ctx),
        llama_n_seq_max(ctx));
    *out = h;
    return JF_LOAD_OK;
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
    for (int32_t i = 0; i < h->n_lora; i++) {
        if (h->loras[i]) {
            llama_adapter_lora_free(h->loras[i]);
        }
    }
    if (h->model) {
        llama_model_free(h->model);
    }
    free(h->cache_tok);
    for (int32_t i = 0; i < JF_CKPT_MAX; i++) {
        free(h->ckpt[i].tok);
        free(h->ckpt[i].data);
    }
    free(h);
}

void jf_backend_free(void) {
    llama_backend_free();
}

int32_t jf_n_seq_max(jf_ctx *h) {
    if (!h) {
        return 0;
    }
    return h->n_seq_max;
}

int32_t jf_n_vocab(jf_ctx *h) {
    if (!h) {
        return 0;
    }
    return h->n_vocab;
}

int32_t jf_n_ctx_seq(jf_ctx *h) {
    if (!h) {
        return 0;
    }
    return h->n_ctx_seq;
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

int32_t jf_token_piece(jf_ctx *h, int32_t id, char *buf, int32_t cap) {
    if (!h || !h->vocab || !buf || cap <= 0 || id < 0 || id >= h->n_vocab) {
        return -1;
    }
    return llama_token_to_piece(h->vocab, id, buf, cap, 0, false);
}

static int32_t *seq_cache(jf_ctx *h, int32_t seq) {
    return h->cache_tok + (size_t) seq * (size_t) h->n_ctx_seq;
}

static void cache_forget(jf_ctx *h) {
    memset(h->cache_len, 0, sizeof(h->cache_len));
}

static int32_t token_prefix(const int32_t *cached, int32_t n_cached, const int32_t *tokens, int32_t n) {
    int32_t i;
    int32_t m = n_cached < n ? n_cached : n;

    for (i = 0; i < m; i++) {
        if (cached[i] != tokens[i]) {
            return i;
        }
    }
    return m;
}

/* Image prefill is not recorded as tokens. Drop it before the next text prefill. */
static void drop_untracked(jf_ctx *h, llama_memory_t mem, int32_t seq) {
    if (h->cache_len[seq] > 0) {
        return;
    }
    if (llama_memory_seq_pos_max(mem, seq) >= 0) {
        llama_memory_seq_rm(mem, seq, 0, -1);
    }
}

/* 0 when the recorded prefix now ends at keep. 1 when the tail cannot be removed. */
static int32_t trim_to(jf_ctx *h, llama_memory_t mem, int32_t seq, int32_t keep) {
    if (h->cache_len[seq] <= keep) {
        return 0;
    }
    if (!llama_memory_seq_rm(mem, seq, keep, -1)) {
        return 1;
    }
    h->cache_len[seq] = keep;
    return 0;
}

static void remember(jf_ctx *h, int32_t seq, const int32_t *tokens, int32_t n) {
    if (n > h->n_ctx_seq) {
        n = h->n_ctx_seq;
    }
    if (n > 0) {
        memcpy(seq_cache(h, seq), tokens, (size_t) n * sizeof(int32_t));
    }
    h->cache_len[seq] = n;
}

/* keep_out is how many leading tokens are already prefilled. Returns 1 if the cache must be reset. */
static int32_t jf_no_reuse(void) {
    const char *value = getenv("BONJEV_NO_REUSE");
    return value && value[0] && value[0] != '0';
}

/* Checkpoint/prefix-restore path can be disabled with BONJEV_CKPT=0. */
static int32_t jf_ckpt_enabled(void) {
    const char *value = getenv("BONJEV_CKPT");
    return !(value && value[0] == '0');
}

static int32_t cached_keep(
    jf_ctx *h,
    llama_memory_t mem,
    int32_t seq,
    const int32_t *tokens,
    int32_t n,
    int reuse,
    int32_t *keep_out) {
    int32_t keep = 0;

    /* Partial trims change the FP reduction order (CUDA is not batch-invariant by
       default), flipping near-ties between runs. BONJEV_NO_REUSE forces a full
       prefill so results do not depend on the request history. */
    if (jf_no_reuse()) {
        reuse = 0;
    }
    drop_untracked(h, mem, seq);
    if (reuse) {
        keep = token_prefix(seq_cache(h, seq), h->cache_len[seq], tokens, n);
        /* The last token is prefilled again so its logits exist. */
        if (keep == n && n > 0) {
            keep = n - 1;
        }
    }
    if (trim_to(h, mem, seq, keep) != 0) {
        return 1;
    }
    *keep_out = keep;
    return 0;
}

static int32_t prefill_one(jf_ctx *h, int32_t seq, const int32_t *tokens, int32_t pos0, int32_t n) {
    int32_t off = 0;

    while (off < n) {
        struct llama_batch batch;
        int32_t m = n - off;
        int32_t i;
        int32_t rc;
        double started = timing_enabled() ? now_ms() : 0.0;

        if (m > h->batch) {
            m = h->batch;
        }
        batch = llama_batch_init(m, 0, 1);
        if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
            llama_batch_free(batch);
            return -1;
        }
        for (i = 0; i < m; i++) {
            batch.token[i] = tokens[off + i];
            batch.pos[i] = pos0 + off + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = seq;
            batch.logits[i] = (off + i == n - 1) ? 1 : 0;
        }
        batch.n_tokens = m;
        rc = llama_decode(h->ctx, batch);
        llama_batch_free(batch);
        if (timing_enabled()) {
            double after_decode = now_ms();
            llama_synchronize(h->ctx);
            fprintf(
                stderr,
                "[bonjev] timing decode seq=%d chunk=%d launch %.1fms sync %.1fms\n",
                seq,
                m,
                after_decode - started,
                now_ms() - after_decode);
        }
        if (rc != 0) {
            return rc;
        }
        off += m;
    }
    return 0;
}

static void kv_reset(jf_ctx *h, llama_memory_t mem, int32_t rc) {
    double started = timing_enabled() ? now_ms() : 0.0;
    fprintf(stderr, "[bonjev] kv reset rc=%d\n", rc);
    llama_memory_clear(mem, true);
    cache_forget(h);
    if (timing_enabled()) {
        fprintf(stderr, "[bonjev] timing kv_reset %.1fms\n", now_ms() - started);
    }
}

/* Record how many leading tokens of each sequence were reused from KV. */
static void note_keep(jf_ctx *h, int32_t nseq, const int32_t *keep, const int32_t *total) {
    int32_t s;

    if (nseq < 0 || nseq > h->n_seq_max) {
        nseq = 0;
    }
    h->last_nseq = nseq;
    for (s = 0; s < nseq; s++) {
        h->last_keep[s] = keep[s];
        h->last_total[s] = total[s];
    }
}

static void note_keep_one(jf_ctx *h, int32_t keep, int32_t total) {
    note_keep(h, 1, &keep, &total);
}

void jf_last_keep(jf_ctx *h, int32_t *nseq_out, int32_t *keep_out, int32_t *total_out, int32_t cap) {
    int32_t s;
    int32_t n = 0;

    if (!h || !nseq_out || !keep_out || !total_out || cap < 0) {
        return;
    }
    n = h->last_nseq;
    if (n > cap) {
        n = cap;
    }
    *nseq_out = n;
    for (s = 0; s < n; s++) {
        keep_out[s] = h->last_keep[s];
        total_out[s] = h->last_total[s];
    }
}

/* Hybrid layers cannot drop a long tail. A saved prefix is restored instead.
   Up to JF_CKPT_MAX prefixes are kept, keyed by their token content, so
   alternating patterns (per-kind routing) do not thrash a single slot. */
static int32_t save_ckpt(jf_ctx *h, int32_t seq, const int32_t *tokens, int32_t n) {
    size_t bytes;
    size_t got;
    uint8_t *buf;
    int32_t slot = -1;
    int32_t i;
    double started = timing_enabled() ? now_ms() : 0.0;

    bytes = llama_state_seq_get_size(h->ctx, seq);
    if (bytes == 0 || n <= 0 || n > h->n_ctx_seq) {
        return -1;
    }
    /* Same prefix: refresh the slot. Otherwise a free slot, else the least
       recently used one. */
    for (i = 0; i < JF_CKPT_MAX; i++) {
        if (h->ckpt[i].n == n && h->ckpt[i].tok &&
            memcmp(h->ckpt[i].tok, tokens, (size_t) n * sizeof(int32_t)) == 0) {
            slot = i;
            break;
        }
    }
    if (slot < 0) {
        for (i = 0; i < JF_CKPT_MAX; i++) {
            if (h->ckpt[i].n == 0) {
                slot = i;
                break;
            }
        }
    }
    if (slot < 0) {
        slot = 0;
        for (i = 1; i < JF_CKPT_MAX; i++) {
            if (h->ckpt[i].used_at < h->ckpt[slot].used_at) {
                slot = i;
            }
        }
    }
    buf = malloc(bytes);
    if (!buf) {
        return -1;
    }
    got = llama_state_seq_get_data(h->ctx, buf, bytes, seq);
    if (got == 0) {
        free(buf);
        return -1;
    }
    if (h->ckpt[slot].n != n || !h->ckpt[slot].tok) {
        int32_t *tok = malloc((size_t) n * sizeof(int32_t));
        if (!tok) {
            free(buf);
            return -1;
        }
        free(h->ckpt[slot].tok);
        h->ckpt[slot].tok = tok;
    }
    memcpy(h->ckpt[slot].tok, tokens, (size_t) n * sizeof(int32_t));
    free(h->ckpt[slot].data);
    h->ckpt[slot].data = buf;
    h->ckpt[slot].size = got;
    h->ckpt[slot].n = n;
    h->ckpt[slot].used_at = ++h->ckpt_clock;
    fprintf(stderr, "[bonjev] prefill checkpoint slot=%d %d tokens, %.1f MiB\n", slot, n, (double) got / (1024.0 * 1024.0));
    if (timing_enabled()) {
        fprintf(stderr, "[bonjev] timing save_ckpt %.1fms\n", now_ms() - started);
    }
    return 0;
}

static int32_t restore_prefix(jf_ctx *h, llama_memory_t mem, int32_t seq, int32_t slot) {
    double started = timing_enabled() ? now_ms() : 0.0;
    llama_memory_seq_rm(mem, seq, 0, -1);
    h->cache_len[seq] = 0;
    if (llama_state_seq_set_data(h->ctx, h->ckpt[slot].data, h->ckpt[slot].size, seq) == 0) {
        return -1;
    }
    remember(h, seq, h->ckpt[slot].tok, h->ckpt[slot].n);
    h->ckpt[slot].used_at = ++h->ckpt_clock;
    if (timing_enabled()) {
        fprintf(stderr, "[bonjev] timing restore_prefix %.1fms\n", now_ms() - started);
    }
    return 0;
}

/* Best checkpoint whose prefix is a proper prefix of `tokens`. */
static int32_t ckpt_find(jf_ctx *h, const int32_t *tokens, int32_t n) {
    int32_t best = -1;
    int32_t i;
    for (i = 0; i < JF_CKPT_MAX; i++) {
        struct jf_ckpt *c = &h->ckpt[i];
        if (c->n > 0 && c->n < n && c->tok && token_prefix(c->tok, c->n, tokens, n) == c->n) {
            if (best < 0 || c->n > h->ckpt[best].n) {
                best = i;
            }
        }
    }
    return best;
}

/* Best checkpoint that is a proper prefix of every sequence. */
static int32_t ckpt_find_many(jf_ctx *h, const int32_t *const *toks, const int32_t *lens, int32_t n_seq) {
    int32_t best = -1;
    int32_t i;
    int32_t s;
    for (i = 0; i < JF_CKPT_MAX; i++) {
        struct jf_ckpt *c = &h->ckpt[i];
        int32_t ok = c->n > 0 && c->tok;
        if (!ok) {
            continue;
        }
        for (s = 0; s < n_seq; s++) {
            if (!(c->n < lens[s] && token_prefix(c->tok, c->n, toks[s], lens[s]) == c->n)) {
                ok = 0;
                break;
            }
        }
        if (ok && (best < 0 || c->n > h->ckpt[best].n)) {
            best = i;
        }
    }
    return best;
}

static int32_t decode_reused(jf_ctx *h, const int32_t *tokens, int32_t n, int reuse) {
    llama_memory_t mem;
    int32_t keep = 0;
    int32_t rc;

    if (n > h->n_ctx_seq) {
        return -1;
    }
    mem = llama_get_memory(h->ctx);
    if (!mem) {
        return -1;
    }
    {
        double t_keep = timing_enabled() ? now_ms() : 0.0;
        if (cached_keep(h, mem, 0, tokens, n, reuse, &keep) != 0) {
            return 1;
        }
        if (timing_enabled()) {
            fprintf(stderr, "[bonjev] timing keep %.1fms\n", now_ms() - t_keep);
        }
    }
    if (keep > 0) {
        fprintf(stderr, "[bonjev] prefill keep %d/%d\n", keep, n);
    }
    {
        double t_pre = timing_enabled() ? now_ms() : 0.0;
        rc = prefill_one(h, 0, tokens + keep, keep, n - keep);
        if (timing_enabled()) {
            fprintf(stderr, "[bonjev] timing prefill %.1fms\n", now_ms() - t_pre);
        }
    }
    if (rc != 0) {
        return rc;
    }
    remember(h, 0, tokens, n);
    note_keep_one(h, keep, n);
    return 0;
}

int32_t jf_decode(jf_ctx *h, const int32_t *tokens, int32_t n) {
    llama_memory_t mem;
    int32_t rc;

    if (!h || !h->ctx || !tokens || n <= 0) {
        return -1;
    }
    if (n > h->n_ctx_seq) {
        return -1;
    }
    mem = llama_get_memory(h->ctx);
    if (!mem) {
        return -1;
    }
    h->last_nseq = 0;
    {
        double t_all = timing_enabled() ? now_ms() : 0.0;
        rc = decode_reused(h, tokens, n, 1);
        if (timing_enabled()) {
            fprintf(stderr, "[bonjev] timing total %.1fms (reuse)\n", now_ms() - t_all);
        }
    }
    if (rc == 0) {
        return 0;
    }
    {
        int32_t slot = jf_ckpt_enabled() ? ckpt_find(h, tokens, n) : -1;
        if (slot >= 0 && restore_prefix(h, mem, 0, slot) == 0) {
            fprintf(stderr, "[bonjev] prefill keep %d/%d\n", h->ckpt[slot].n, n);
            rc = prefill_one(h, 0, tokens + h->ckpt[slot].n, h->ckpt[slot].n, n - h->ckpt[slot].n);
            if (rc == 0) {
                remember(h, 0, tokens, n);
                note_keep_one(h, h->ckpt[slot].n, n);
                return 0;
            }
        }
    }
    {
        int32_t shared = token_prefix(seq_cache(h, 0), h->cache_len[0], tokens, n);
        if (shared == n && n > 0) {
            shared = n - 1;
        }
        kv_reset(h, mem, rc);
        /* A prefix shorter than a system prompt is not worth a 150 MiB
           checkpoint slot: prefill it and continue. */
        if (shared >= 16 && shared < n) {
            rc = prefill_one(h, 0, tokens, 0, shared);
            if (rc == 0) {
                save_ckpt(h, 0, tokens, shared);
                rc = prefill_one(h, 0, tokens + shared, shared, n - shared);
            }
            if (rc == 0) {
                remember(h, 0, tokens, n);
                note_keep_one(h, 0, n);
                return 0;
            }
            kv_reset(h, mem, rc);
            return rc;
        }
    }
    rc = decode_reused(h, tokens, n, 0);
    if (rc != 0) {
        kv_reset(h, mem, rc);
    }
    return rc;
}

static int32_t common_prefix(const int32_t *const *toks, const int32_t *lens, int32_t n_seq) {
    int32_t n = lens[0];
    int32_t i;
    int32_t s;

    for (s = 1; s < n_seq; s++) {
        if (lens[s] < n) {
            n = lens[s];
        }
    }
    for (i = 0; i < n; i++) {
        int32_t tok = toks[0][i];
        for (s = 1; s < n_seq; s++) {
            if (toks[s][i] != tok) {
                return i;
            }
        }
    }
    return n;
}

/* One token, many sequence ids. pos0 is the position of tokens[0]. */
static int32_t decode_shared_prefix(
    jf_ctx *h, const int32_t *tokens, int32_t pos0, int32_t n, int32_t n_seq) {
    int32_t off = 0;

    while (off < n) {
        struct llama_batch batch;
        int32_t m = n - off;
        int32_t i;
        int32_t s;
        int32_t rc;

        if (m > h->batch) {
            m = h->batch;
        }
        batch = llama_batch_init(m, 0, n_seq);
        if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
            llama_batch_free(batch);
            return -1;
        }
        for (i = 0; i < m; i++) {
            batch.token[i] = tokens[off + i];
            batch.pos[i] = pos0 + off + i;
            batch.n_seq_id[i] = n_seq;
            for (s = 0; s < n_seq; s++) {
                batch.seq_id[i][s] = s;
            }
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

static int32_t decode_many_reused(
    jf_ctx *h, const int32_t *const *toks, const int32_t *lens, int32_t n_seq, float *out_logits, int reuse) {
    llama_memory_t mem;
    int32_t keep[JF_SEQ_MAX];
    int32_t pos[JF_SEQ_MAX];
    int32_t min_keep;
    int32_t max_keep;
    int32_t common;
    int32_t width;
    int32_t max_n;
    int32_t s;
    int32_t rc;

    mem = llama_get_memory(h->ctx);
    if (!mem) {
        return -1;
    }
    if (jf_no_reuse()) {
        reuse = 0;
    }
    width = h->batch / n_seq;
    if (width < 1) {
        return -1;
    }
    max_n = 0;
    for (s = 0; s < n_seq; s++) {
        if (lens[s] > max_n) {
            max_n = lens[s];
        }
        if (lens[s] > h->n_ctx_seq) {
            return -1;
        }
    }
    min_keep = 0;
    max_keep = 0;
    {
        int32_t restored = 0;
        if (reuse && jf_ckpt_enabled()) {
            int32_t slot = ckpt_find_many(h, toks, lens, n_seq);
            if (slot >= 0) {
                restored = 1;
                for (s = 0; s < n_seq; s++) {
                    if (restore_prefix(h, mem, s, slot) != 0) {
                        return 1;
                    }
                    keep[s] = h->ckpt[slot].n;
                }
                min_keep = h->ckpt[slot].n;
                max_keep = h->ckpt[slot].n;
            }
        }
        if (!restored) {
            for (s = 0; s < n_seq; s++) {
                if (cached_keep(h, mem, s, toks[s], lens[s], reuse, &keep[s]) != 0) {
                    return 1;
                }
                if (s == 0 || keep[s] < min_keep) {
                    min_keep = keep[s];
                }
                if (keep[s] > max_keep) {
                    max_keep = keep[s];
                }
            }
        }
    }
    /* Reused-from-KV counts, before the shared-prefix block rewrites keep[]. */
    note_keep(h, n_seq, keep, lens);
    common = n_seq > 1 ? common_prefix(toks, lens, n_seq) : 0;
    if (min_keep == max_keep && common > min_keep) {
        int32_t ended = 0;
        rc = decode_shared_prefix(h, toks[0] + min_keep, min_keep, common - min_keep, n_seq);
        if (rc != 0) {
            return rc;
        }
        for (s = 0; s < n_seq; s++) {
            if (lens[s] == common) {
                ended = 1;
                break;
            }
        }
        if (ended) {
            const float *row = llama_get_logits(h->ctx);
            if (!row) {
                return -1;
            }
            for (s = 0; s < n_seq; s++) {
                if (lens[s] == common) {
                    memcpy(
                        out_logits + (size_t) s * (size_t) h->n_vocab,
                        row,
                        (size_t) h->n_vocab * sizeof(float));
                }
            }
        }
        if (min_keep > 0) {
            fprintf(stderr, "[bonjev] prefill keep %d/%d\n", min_keep, common);
        }
        for (s = 0; s < n_seq; s++) {
            keep[s] = common;
        }
    } else if (min_keep > 0) {
        fprintf(stderr, "[bonjev] prefill keep %d/%d\n", min_keep, max_n);
    }

    for (s = 0; s < n_seq; s++) {
        pos[s] = keep[s];
    }
    for (;;) {
        struct llama_batch batch;
        int32_t live = 0;
        int32_t count = 0;
        int32_t t = 0;
        int32_t n_out = 0;
        int32_t out_seq[JF_SEQ_MAX];

        for (s = 0; s < n_seq; s++) {
            int32_t m;
            if (pos[s] >= lens[s]) {
                continue;
            }
            live = 1;
            m = lens[s] - pos[s];
            if (m > width) {
                m = width;
            }
            count += m;
        }
        if (!live) {
            break;
        }
        batch = llama_batch_init(count, 0, 1);
        if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
            llama_batch_free(batch);
            return -1;
        }
        for (s = 0; s < n_seq; s++) {
            int32_t m;
            int32_t j;
            if (pos[s] >= lens[s]) {
                continue;
            }
            m = lens[s] - pos[s];
            if (m > width) {
                m = width;
            }
            for (j = 0; j < m; j++) {
                int32_t at = pos[s] + j;
                batch.token[t] = toks[s][at];
                batch.pos[t] = at;
                batch.n_seq_id[t] = 1;
                batch.seq_id[t][0] = s;
                batch.logits[t] = (at == lens[s] - 1) ? 1 : 0;
                if (batch.logits[t]) {
                    out_seq[n_out] = s;
                    n_out++;
                }
                t++;
            }
            pos[s] += m;
        }
        batch.n_tokens = t;
        rc = llama_decode(h->ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            return rc;
        }
        if (n_out > 0) {
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
    }
    for (s = 0; s < n_seq; s++) {
        remember(h, s, toks[s], lens[s]);
    }
    return 0;
}

int32_t jf_decode_many(
    jf_ctx *h, const int32_t *const *toks, const int32_t *lens, int32_t n_seq, float *out_logits) {
    llama_memory_t mem;
    int32_t s;
    int32_t rc;

    if (!h || !h->ctx || !toks || !lens || !out_logits || n_seq < 1 || n_seq > h->n_seq_max) {
        return -1;
    }
    for (s = 0; s < n_seq; s++) {
        if (!toks[s] || lens[s] <= 0) {
            return -1;
        }
    }
    mem = llama_get_memory(h->ctx);
    if (!mem) {
        return -1;
    }
    h->last_nseq = 0;
    rc = decode_many_reused(h, toks, lens, n_seq, out_logits, 1);
    if (rc == 0) {
        return 0;
    }
    kv_reset(h, mem, rc);
    rc = decode_many_reused(h, toks, lens, n_seq, out_logits, 0);
    if (rc != 0) {
        kv_reset(h, mem, rc);
    }
    return rc;
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
    int32_t seq;
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
    /* The image is not a continuation of the recorded text prefix. */
    for (seq = 0; seq < h->n_seq_max; seq++) {
        if (h->cache_len[seq] > 0 || llama_memory_seq_pos_max(mem, seq) >= 0) {
            llama_memory_seq_rm(mem, seq, 0, -1);
            h->cache_len[seq] = 0;
        }
    }
    n_past = 0;
    rc = mtmd_helper_eval_chunks(h->mtmd, h->ctx, chunks, 0, 0, h->batch, true, &n_past);
    if (rc != 0) {
        kv_reset(h, mem, rc);
        n_past = 0;
        rc = mtmd_helper_eval_chunks(h->mtmd, h->ctx, chunks, 0, 0, h->batch, true, &n_past);
        if (rc != 0) {
            kv_reset(h, mem, rc);
        }
    }
    n_tok = mtmd_helper_get_n_tokens(chunks);
    if (n_tokens_out) {
        *n_tokens_out = n_tok > (size_t) INT32_MAX ? INT32_MAX : (int32_t) n_tok;
    }
    if (rc == 0) {
        int32_t total = n_tok > (size_t) INT32_MAX ? INT32_MAX : (int32_t) n_tok;
        note_keep_one(h, 0, total);
    } else {
        h->last_nseq = 0;
    }
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bitmap);
    return rc;
}
