#!/usr/bin/env python3
"""Extract the Clef post-training delta from Cloudflare/clef vs Qwen/Qwen3.8-27B.

For every tensor present in both releases:
  - diff in fp32; bit-identical tensors are skipped (untouched)
  - 2D weights: randomized SVD to rank R, save lora_A=(R,in), lora_B=(out,R)
  - other tensors: metrics only (not representable as LoRA)
Text parts -> parts_text/, vision parts -> parts_visual/ (mmproj naming).
Outputs: metrics.jsonl + summary.json in OUT.

Env:
  CLEF_ONLY_LAYERS="0-3,5"  restrict to layer numbers (for partial downloads)
"""
import glob, gc, json, os, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

HUB = os.path.expanduser('~/.cache/huggingface/hub')
QWEN = os.environ.get('CLEF_QWEN_DIR') or glob.glob(f'{HUB}/models--Qwen--Qwen3.8-27B/snapshots/*')[0]
CLEF = os.environ.get('CLEF_CLEF_DIR') or glob.glob(f'{HUB}/models--Cloudflare--clef/snapshots/*')[0]
TMP = os.environ.get('TMPDIR', '/tmp')
QIDX = os.environ.get('CLEF_QWEN_IDX', f'{TMP}/bonjev_qwen_idx.json')
CIDX = os.environ.get('CLEF_CLEF_IDX', f'{TMP}/bonjev_clef_idx.json')
OUT = os.environ.get('CLEF_OUT', 'work/deltas/clef-27b')
PARTS_TEXT = f'{OUT}/parts_text'
PARTS_VIS = f'{OUT}/parts_visual'
R = 256
OVER = 96
ITERS = 2

os.makedirs(PARTS_TEXT, exist_ok=True)
os.makedirs(PARTS_VIS, exist_ok=True)
torch.set_num_threads(int(os.environ.get('CLEF_THREADS', '4')))

q_idx = json.load(open(QIDX))['weight_map']
c_idx = json.load(open(CIDX))['weight_map']
keys = [k for k in c_idx if not k.startswith('mtp.')]

only = os.environ.get('CLEF_ONLY_LAYERS')
if only:
    wanted = set()
    for part in only.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            wanted.update(range(int(a), int(b) + 1))
        elif part:
            wanted.add(int(part))
    def layer_of(k):
        if k.startswith('model.language_model.layers.'):
            return int(k.split('.')[3])
        if k.startswith('model.visual.blocks.'):
            return int(k.split('.')[3])
        return -1
    keys = [k for k in keys if layer_of(k) in wanted]

if os.environ.get('CLEF_NO_VIS'):
    keys = [k for k in keys if not k.startswith('model.visual.')]

# Process shard by shard, resumable: done keys are skipped, chunk size bounds each run.
keys.sort(key=lambda k: (c_idx[k], q_idx[k], k))
done_path = f'{OUT}/done.txt'
done = set()
if os.path.exists(done_path):
    done = set(open(done_path).read().split())
keys = [k for k in keys if k not in done]
max_keys = int(os.environ.get('CLEF_MAX_KEYS', '0'))
if max_keys > 0:
    keys = keys[:max_keys]
print(f'{len(keys)} tensors to check (done so far: {len(done)})', flush=True)

handles = {}
class MissingShard(Exception):
    pass

def close_handles():
    for h in handles.values():
        try:
            h.__exit__(None, None, None)
        except Exception:
            pass
    handles.clear()

def tensor(root, wm, key):
    shard = wm[key]
    p = os.path.join(root, shard)
    if not os.path.isfile(p):
        raise MissingShard(p)
    h = handles.get((root, shard))
    if h is None:
        h = safe_open(p, 'pt')
        handles[(root, shard)] = h
    return h.get_tensor(key)

def rss_gib():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return 0.0

def rsvd(A, r, over=OVER, iters=ITERS):
    m, n = A.shape
    r_eff = min(r, min(m, n))
    k = min(r_eff + over, min(m, n))
    G = torch.randn(n, k, dtype=A.dtype)
    Y = A @ G
    for _ in range(iters):
        Q, _ = torch.linalg.qr(Y)
        Z, _ = torch.linalg.qr(A.T @ Q)
        Y = A @ Z
    Q, _ = torch.linalg.qr(Y)
    B = Q.T @ A
    Uhat, S, Vh = torch.linalg.svd(B, full_matrices=False)
    return (Q @ Uhat)[:, :r_eff], S[:r_eff], Vh[:r_eff]

def gguf_name(key):
    if key.startswith('model.language_model.layers.'):
        rest = key[len('model.language_model.layers.'):]
        n, _, t = rest.partition('.')
        base = f'blk.{n}.'
        if t.startswith('self_attn.'):
            m = {'q_proj.weight': 'attn_q.weight', 'k_proj.weight': 'attn_k.weight',
                 'v_proj.weight': 'attn_v.weight', 'o_proj.weight': 'attn_output.weight',
                 'q_norm.weight': 'attn_q_norm.weight', 'k_norm.weight': 'attn_k_norm.weight'}
            s = t[len('self_attn.'):]
            return base + m[s] if s in m else None
        if t.startswith('linear_attn.'):
            m = {'in_proj_qkv.weight': 'attn_qkv.weight', 'in_proj_z.weight': 'attn_gate.weight',
                 'out_proj.weight': 'ssm_out.weight', 'conv1d.weight': 'ssm_conv1d.weight',
                 'in_proj_a.weight': 'ssm_alpha.weight', 'in_proj_b.weight': 'ssm_beta.weight',
                 'A_log': 'ssm_a', 'dt_bias': 'ssm_dt.bias', 'norm.weight': 'ssm_norm.weight'}
            s = t[len('linear_attn.'):]
            return base + m[s] if s in m else None
        if t.startswith('mlp.'):
            m = {'gate_proj.weight': 'ffn_gate.weight', 'up_proj.weight': 'ffn_up.weight',
                 'down_proj.weight': 'ffn_down.weight'}
            s = t[len('mlp.'):]
            return base + m[s] if s in m else None
        if t == 'input_layernorm.weight':
            return base + 'attn_norm.weight'
        if t == 'post_attention_layernorm.weight':
            return base + 'post_attention_norm.weight'
        return None
    if key.startswith('model.visual.blocks.'):
        rest = key[len('model.visual.blocks.'):]
        n, _, t = rest.partition('.')
        m = {'attn.qkv.weight': 'attn_qkv.weight', 'attn.proj.weight': 'attn_out.weight',
             'mlp.linear_fc1.weight': 'ffn_up.weight', 'mlp.linear_fc2.weight': 'ffn_down.weight'}
        return f'v.blk.{n}.' + m[t] if t in m else None
    if key == 'model.visual.merger.linear_fc1.weight':
        return 'mm.0.weight'
    if key == 'model.visual.merger.linear_fc2.weight':
        return 'mm.2.weight'
    if key == 'lm_head.weight':
        return 'output.weight'
    if key == 'model.language_model.embed_tokens.weight':
        return 'token_embd.weight'
    return None

def category(key):
    if key.startswith('model.visual.'):
        return 'vision'
    if key.startswith('model.language_model.layers.'):
        rest = key[len('model.language_model.layers.'):]
        rest = rest.split('.', 1)[1] if '.' in rest else ''
        for p, name in [('self_attn.', 'full_attn'), ('linear_attn.', 'linear_attn'), ('mlp.', 'mlp')]:
            if rest.startswith(p):
                return name
        return 'layer_norm_or_other'
    if key.startswith('lm_head') or key.startswith('model.language_model.embed'):
        return 'head_or_embed'
    return 'other'

metrics_path = f'{OUT}/metrics.jsonl'
mf = open(metrics_path, 'a')
donef = open(done_path, 'a')
summary = {'total': 0, 'identical': 0, 'changed_2d': 0, 'changed_other': 0,
           'no_gguf_name': [], 'by_category': {}, 'unchanged_by_category': {}}
t0 = time.time()
cur_shard = None

for i, key in enumerate(keys):
    shard = c_idx[key]
    if shard != cur_shard:
        close_handles()   # release mmaps of the previous shard pair
        cur_shard = shard
    try:
        qt = tensor(QWEN, q_idx, key)
        ct = tensor(CLEF, c_idx, key)
    except MissingShard as e:
        print(f'skip (shard not ready): {key} [{e}]', flush=True)
        continue
    summary['total'] += 1
    cat = category(key)
    if qt.shape != ct.shape:
        print(f'SHAPE MISMATCH {key}: {tuple(qt.shape)} vs {tuple(ct.shape)}', flush=True)
        donef.write(key + '\n'); donef.flush()
        continue
    d = ct.float() - qt.float()
    max_abs = d.abs().max().item()
    if max_abs == 0.0:
        summary['identical'] += 1
        summary['unchanged_by_category'][cat] = summary['unchanged_by_category'].get(cat, 0) + 1
        donef.write(key + '\n'); donef.flush()
        del qt, ct, d
        continue
    gname = gguf_name(key)
    rec = {'name': key, 'gguf': gname, 'cat': cat, 'shape': list(d.shape),
           'max_abs': max_abs, 'norm': d.norm().item()}
    if d.ndim == 2:
        wnorm = qt.float().norm().item()
        rec['rel'] = d.norm().item() / max(wnorm, 1e-9)
        r_eff = min(R, min(d.shape))
        U, S, Vh = rsvd(d, r_eff)
        total_energy = d.norm().item() ** 2
        rec['top_energy'] = (S ** 2).sum().item() / total_energy
        for k in (32, 64, 128):
            rec[f'E{k}'] = (S[:k] ** 2).sum().item() / total_energy
        rec['S_head'] = S[:4].tolist()
        rec['S_tail'] = S[-4:].tolist()
        A = (S.sqrt().unsqueeze(1) * Vh).to(torch.float16).contiguous()   # (r, in)
        B = (U * S.sqrt().unsqueeze(0)).to(torch.float16).contiguous()   # (out, r)
        rec['in'] = d.shape[1]
        rec['out'] = d.shape[0]
        rec['rank'] = r_eff
        if gname:
            vis = gname.startswith('v.') or gname.startswith('mm.')
            save_file({'lora_a': A, 'lora_b': B},
                      f'{PARTS_VIS if vis else PARTS_TEXT}/{gname}.safetensors')
            rec['saved'] = True
            summary['changed_2d'] += 1
        else:
            rec['saved'] = False
            summary['no_gguf_name'].append(key)
        print(f'[{i}/{len(keys)}] {cat:16s} {key} {tuple(d.shape)} rel={rec["rel"]:.4f} '
              f'E={rec["top_energy"]:.4f} saved={rec.get("saved")} rss={rss_gib():.1f}G', flush=True)
        del U, S, Vh, A, B
    else:
        summary['changed_other'] += 1
        print(f'[{i}/{len(keys)}] {cat:16s} {key} shape={tuple(d.shape)} non-2D max_abs={max_abs:.3e}', flush=True)
    summary['by_category'][cat] = summary['by_category'].get(cat, 0) + 1
    mf.write(json.dumps(rec) + '\n')
    mf.flush()
    donef.write(key + '\n'); donef.flush()
    del qt, ct, d
    if i % 4 == 0:
        gc.collect()

mf.close()
donef.close()
summary['seconds'] = time.time() - t0
suffix = only.replace(',', '_') if only else 'all'
json.dump(summary, open(f'{OUT}/summary_{suffix}.json', 'w'), indent=1)
print(json.dumps(summary, indent=1)[:2500], flush=True)
