#!/usr/bin/env python3
"""Direct text generation check (no decision readout): does the model produce
coherent text or a garbled mess, with and without a LoRA?

Starts llama-server (prism fork) with the model + optional LoRA, sends N prompts
to /completion, and reports simple garbling metrics + samples.

Usage: direct_infer.py <model_gguf> <lora|-> <port> [n_prompts]
"""
import json, subprocess, sys, time, urllib.request, os, re, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _prism_bin():
    d = os.environ.get("PRISM_LLAMA_DIR")
    if not d:
        p = os.path.join(ROOT, "tooling", "prism-lib-dir")
        if os.path.isfile(p):
            d = open(p).read().strip()
    if not d:
        raise SystemExit("set PRISM_LLAMA_DIR (or run scripts/build-prism-llama.sh)")
    return d


BIN = _prism_bin()

PROMPTS = [
    "The capital of France is", "2 + 2 =", "Write a haiku about the sea:\n",
    "The chemical symbol for gold is", "Once upon a time,", "The largest planet in the solar system is",
    "Translate 'hello' to Spanish:", "The first president of the United States was",
    "What is the boiling point of water?", "List three colors:", "The opposite of hot is",
    "A noun is a word that", "The sun rises in the", "Water freezes at",
    "The author of Romeo and Juliet is", "How many days are in a week?",
    "The speed of light is approximately", "Name a mammal:", "The past tense of 'go' is",
    "The capital of Japan is", "Three prime numbers are", "The color of the sky on a clear day is",
    "What is 10 minus 4?", "A triangle has how many sides?", "The plural of 'child' is",
    "The longest river in the world is", "What do bees make?", "The freezing point of water in Celsius is",
    "The square root of 16 is", "Name a fruit:", "The opposite of up is", "A baby dog is called a",
    "The chemical formula for water is", "How many months are in a year?", "The capital of Italy is",
    "What sound does a cat make?", "The largest ocean is the", "Write a short greeting:",
    "The number after 9 is", "The color of grass is", "A group of wolves is called a",
    "The tallest mountain is", "What is 3 times 3?", "The past tense of 'eat' is",
    "The capital of Germany is", "Name a day of the week:", "The opposite of big is",
    "How many hours are in a day?", "The chemical symbol for oxygen is", "A baby cat is called a",
]


def nonprint_ratio(s):
    if not s:
        return 1.0
    bad = sum(1 for ch in s if not (ch.isprintable() or ch in "\n\t"))
    return bad / len(s)


def max_repeat_run(s):
    """Longest run of the same repeated token (word or 2-gram)."""
    toks = s.split()
    best = 0
    for n in (1, 2, 3):
        i = 0
        while i < len(toks):
            j = i
            while j + n <= len(toks) and toks[j:j + n] == toks[i:i + n]:
                j += n
            best = max(best, (j - i) // n)
            i += 1
    return best


def uniq_ratio(s):
    toks = s.split()
    return len(set(toks)) / len(toks) if toks else 0.0


def start_server(model, lora, port, log):
    env = dict(os.environ, LD_LIBRARY_PATH=BIN)
    cmd = [f"{BIN}/llama-server", "-m", model, "--port", str(port), "-c", "4096", "-ngl", "99", "--temp", "0"]
    if lora and lora != "-":
        cmd += ["--lora", lora]
    f = open(log, "w")
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return p
        except Exception:
            time.sleep(1)
    p.kill()
    raise RuntimeError("server did not start")


def complete(port, prompt, n=40):
    body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0, "top_k": 1, "cache_prompt": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read()).get("content", "")


def main():
    model, lora, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    label = (os.path.basename(model) + " + " + (os.path.basename(lora) if lora != "-" else "BASE"))
    outdir = os.path.join(os.environ.get("TMPDIR", "/tmp"), "bonjev", "direct")
    os.makedirs(outdir, exist_ok=True)
    log = f"{outdir}/{re.sub(r'[^A-Za-z0-9]+','_',label)}.serve.log"
    p = start_server(model, lora, port, log)
    outs = []
    try:
        for pr in PROMPTS[:n]:
            try:
                outs.append(complete(port, pr))
            except Exception as e:
                outs.append(f"<ERR {e}>")
    finally:
        p.terminate()
        try:
            p.wait(timeout=10)
        except Exception:
            p.kill()
    empty = sum(1 for o in outs if not o.strip())
    garbled = sum(1 for o in outs if nonprint_ratio(o) > 0.02)
    rep = sum(1 for o in outs if max_repeat_run(o) >= 6)
    ur = sum(uniq_ratio(o) for o in outs) / len(outs)
    print(f"== {label}: n={len(outs)} empty={empty} garbled={garbled} rep>=6={rep} uniq_word_ratio={ur:.2f}")
    for o in outs[:4]:
        print("   |", o.strip().replace("\n", " ")[:90])


if __name__ == "__main__":
    main()
