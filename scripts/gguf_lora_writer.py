#!/usr/bin/env python3
"""Streaming GGUF LoRA adapter writer. Never holds more than one tensor in RAM.

Part files are safetensors with lora_a (r, in) and lora_b (out, r), fp16/fp32.
All tensors are padded to a uniform rank R and adapter.lora.alpha is set to R,
so llama.cpp computes scale = alpha / rank = 1 for every tensor.

Usage: gguf_lora_writer.py <parts_dir> <out.gguf> [rank] [arch]
"""
import glob, os, struct, sys
import numpy as np
from safetensors import safe_open
from gguf.constants import GGMLQuantizationType
from gguf.quants import quantize

GGML_F16 = 1
GGML_Q8_0 = 8
GGML_Q4_0 = 2
ALIGN = 32


def s(v):
    b = v.encode('utf-8')
    return struct.pack('<Q', len(b)) + b


def kv_str(key, value):
    return s(key) + struct.pack('<I', 8) + s(value)


def kv_f32(key, value):
    return s(key) + struct.pack('<I', 6) + struct.pack('<f', value)


class StreamingAdapter:
    def __init__(self, path, arch, alpha, rank, outtype='f16'):
        self.path = path
        self.arch = arch
        self.alpha = alpha
        self.rank = rank
        self.outtype = outtype
        self.gtype = {'f16': GGML_F16, 'q8_0': GGML_Q8_0, 'q4_0': GGML_Q4_0}[outtype]
        self.items = []   # (name, ne_tuple, type, source_path, kind)
        self.nbytes = 0

    def add_part(self, name, part_path, a_shape, b_shape):
        """a_shape=(r,in), b_shape=(out,r) as stored (numpy order)."""
        ne_a = (a_shape[1], self.rank)   # GGUF ne: (in, R)
        ne_b = (self.rank, b_shape[0])   # GGUF ne: (R, out)
        self.items.append((name + '.lora_a', ne_a, part_path, 'a'))
        self.items.append((name + '.lora_b', ne_b, part_path, 'b'))

    def _size(self, ne):
        n = 1
        for d in ne:
            n *= d
        if self.gtype == GGML_F16:
            return n * 2
        if self.gtype == GGML_Q8_0:
            return (n // 32) * 34
        return (n // 32) * 18   # Q4_0

    def write(self):
        # pass 1: compute offsets (each tensor aligned to ALIGN)
        offsets = []
        off = 0
        for name, ne, _, _ in self.items:
            off = (off + ALIGN - 1) // ALIGN * ALIGN
            offsets.append(off)
            off += self._size(ne)

        header = bytearray()
        header += b'GGUF' + struct.pack('<I', 3)
        header += struct.pack('<Q', len(self.items))
        n_kv = 4
        header += struct.pack('<Q', n_kv)
        header += kv_str('general.architecture', self.arch)
        header += kv_str('general.type', 'adapter')
        header += kv_str('adapter.type', 'lora')
        header += kv_f32('adapter.lora.alpha', float(self.alpha))
        for (name, ne, _, _), o in zip(self.items, offsets):
            header += s(name)
            header += struct.pack('<I', len(ne))
            for d in ne:
                header += struct.pack('<Q', d)
            header += struct.pack('<I', self.gtype)
            header += struct.pack('<Q', o)
        pad = (-len(header)) % ALIGN
        header += b'\x00' * pad
        data_start = len(header)

        handles = {}
        tmp = self.path + '.tmp'
        with open(tmp, 'wb') as f:
            f.write(header)
            for (name, ne, part, kind), o in zip(self.items, offsets):
                f.seek(data_start + o)
                h = handles.get(part)
                if h is None:
                    h = safe_open(part, 'pt')
                    handles[part] = h
                t = h.get_tensor('lora_a' if kind == 'a' else 'lora_b').float().numpy()
                # truncate rank (rows of A / cols of B, ordered by descending S)
                if kind == 'a' and t.shape[0] > self.rank:
                    t = t[:self.rank]
                if kind == 'b' and t.shape[1] > self.rank:
                    t = t[:, :self.rank]
                # pad rank to self.rank with zeros
                if kind == 'a' and t.shape[0] < self.rank:
                    t = np.vstack([t, np.zeros((self.rank - t.shape[0], t.shape[1]), dtype=t.dtype)])
                if kind == 'b' and t.shape[1] < self.rank:
                    t = np.hstack([t, np.zeros((t.shape[0], self.rank - t.shape[1]), dtype=t.dtype)])
                assert (t.shape[1], t.shape[0]) == ne, (name, t.shape, ne)
                if self.gtype == GGML_F16:
                    raw = t.astype(np.float16).tobytes()
                else:
                    qtype = GGMLQuantizationType.Q8_0 if self.gtype == GGML_Q8_0 else GGMLQuantizationType.Q4_0
                    raw = quantize(np.ascontiguousarray(t, dtype=np.float32), qtype).tobytes()
                assert len(raw) == self._size(ne), (name, len(raw), self._size(ne))
                f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
        return self.path


def build(parts_dir, out, rank=256, arch='qwen35', outtype='f16', shift=0):
    import re
    w = StreamingAdapter(out, arch, alpha=rank, rank=rank, outtype=outtype)
    n = 0
    for p in sorted(glob.glob(f'{parts_dir}/*.safetensors')):
        name = os.path.basename(p)[:-len('.safetensors')]
        if shift:
            m = re.match(r'blk\.(\d+)\.(.*)$', name)
            if m:
                name = f'blk.{int(m.group(1)) + shift}.{m.group(2)}'
        if name in ('output.weight', 'token_embd.weight'):
            print('skip', name)
            continue
        with safe_open(p, 'pt') as h:
            a_shape = tuple(h.get_slice('lora_a').get_shape())
            b_shape = tuple(h.get_slice('lora_b').get_shape())
        w.add_part(name, p, a_shape, b_shape)
        n += 1
    w.write()
    size = os.path.getsize(out)
    print(f'wrote {out}: {n} tensors, rank={rank}, alpha={rank}, arch={arch}, outtype={outtype}, shift={shift}, size={size/2**20:.1f} MiB')
    return out


if __name__ == '__main__':
    parts, out = sys.argv[1], sys.argv[2]
    rank = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    arch = sys.argv[4] if len(sys.argv) > 4 else 'qwen35'
    outtype = sys.argv[5] if len(sys.argv) > 5 else 'f16'
    shift = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    build(parts, out, rank, arch, outtype, shift)
