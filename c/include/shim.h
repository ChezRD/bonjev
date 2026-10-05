#ifndef JF_SHIM_H
#define JF_SHIM_H

#include <stdint.h>

typedef struct jf_ctx jf_ctx;

/* Load ggml plugins from `dir`. `dir` may be NULL. */
void jf_load_backends(const char *dir);

/* Load status codes. */
enum {
    JF_LOAD_OK = 0,
    JF_LOAD_MODEL = 1,    /* weights failed (out of memory or a bad file) */
    JF_LOAD_CONTEXT = 2,  /* context init failed (usually out of memory) */
    JF_LOAD_LORA = 3,     /* an adapter failed to load */
    JF_LOAD_ARGS = 4,     /* bad arguments */
    JF_LOAD_INTERNAL = 5  /* allocation or shape failure */
};

/* Loads into *out and returns JF_LOAD_OK on success. On failure returns a
   nonzero code and leaves *out NULL.
   `lora` may be NULL; otherwise `<path>[,scale][;<path>[,scale]...]` attaches up
   to JF_LORA_MAX adapters. The context has 3 sequences and a unified KV the size
   of n_ctx. */
int32_t jf_load(
    const char *path,
    const char *lora,
    int32_t n_ctx,
    int32_t n_threads,
    int32_t n_gpu_layers,
    jf_ctx **out);

void jf_free(jf_ctx *h);

/* Process-lifetime backend. Not tied to a single jf_ctx. */
void jf_backend_free(void);

int32_t jf_n_vocab(jf_ctx *h);
int32_t jf_n_seq_max(jf_ctx *h);

/* Max prompt tokens per sequence for this context (llama_n_ctx_seq). */
int32_t jf_n_ctx_seq(jf_ctx *h);

/* Same return contract as llama_tokenize, including INT32_MIN on overflow. */
int32_t jf_tokenize(jf_ctx *h, const char *text, int32_t add_special, int32_t *out, int32_t cap);

/* Detokenize one token id into `buf` (no lstrip, no special handling).
   Returns the byte count, or the negated needed size when `cap` is too small. */
int32_t jf_token_piece(jf_ctx *h, int32_t id, char *buf, int32_t cap);

/* 0 on success. Logits for the last token are then available from jf_logits. */
int32_t jf_decode(jf_ctx *h, const int32_t *tokens, int32_t n);

/* Decode n_seq independent prompts. out_logits is n_seq * n_vocab, row per sequence.
   0 on success. Sequences must be non-empty and n_seq must fit the context. */
int32_t jf_decode_many(
    jf_ctx *h,
    const int32_t *const *toks,
    const int32_t *lens,
    int32_t n_seq,
    float *out_logits);

/* NULL if logits were not produced. Points at n_vocab floats. */
const float *jf_logits(jf_ctx *h);

/* Tokens reused from KV vs prompt length for the last decode call.
   Writes up to `cap` entries: keep_out[i]/total_out[i] per sequence, nseq_out count.
   No-op on bad arguments. Stale after a failed call is cleared to zero sequences. */
void jf_last_keep(jf_ctx *h, int32_t *nseq_out, int32_t *keep_out, int32_t *total_out, int32_t cap);

/* Load the vision projector. 0 on success. Text decode keeps working if this is never called. */
int32_t jf_vision_load(jf_ctx *h, const char *mmproj, int32_t n_threads);

/* Marker the prompt must contain once, or NULL when no projector is loaded. */
const char *jf_media_marker(jf_ctx *h);

/* Decode one RGB image (nx * ny * 3 bytes, row-major) inside `prompt`.
   `prompt` must contain the media marker once. 0 on success.
   *n_tokens_out receives the token count of the mixed prompt when non-NULL. */
int32_t jf_decode_rgb(
    jf_ctx *h,
    const char *prompt,
    const unsigned char *rgb,
    uint32_t nx,
    uint32_t ny,
    int32_t *n_tokens_out);

#endif
