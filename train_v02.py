"""Train RC Lilguy v0.2: a CPU-first multi-order context-cache language model.

This is an experimental alternative to v0.1's neural Transformer. It learns
next-character distributions from 1- through 5-character contexts in one pass.
It is fast and compact, but it is not a semantic reasoning model.
"""
import argparse, json, random, time
from collections import Counter, defaultdict
from pathlib import Path
from train import fetch_stories

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "checkpoints"
MODEL_ID = "RC-Lilguy-v0.2"
MODEL_NAME = "RC Lilguy v0.2 — Context Cache"

def train_cache(text, max_order=5):
    tables = [defaultdict(Counter) for _ in range(max_order + 1)]
    # Each character target is indexed by the suffix immediately before it.
    for i, target in enumerate(text):
        for order in range(1, min(max_order, i) + 1):
            tables[order][text[i-order:i]][target] += 1
        tables[0][""][target] += 1
    # Convert Counter objects into compact, JSON-serializable lists.
    packed = []
    for table in tables:
        packed.append({ctx: [[ch, n] for ch, n in counts.most_common()]
                       for ctx, counts in table.items()})
    return packed

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stories", type=int, default=800)
    p.add_argument("--max-order", type=int, default=5)
    args = p.parse_args()
    random.seed(42)
    start = time.time()
    text = fetch_stories(args.stories)
    print(f"Building context tables from {len(text):,} characters…", flush=True)
    tables = train_cache(text, max(1, min(5, args.max_order)))
    OUT.mkdir(exist_ok=True)
    model = {
        "model_id": MODEL_ID, "model_name": MODEL_NAME,
        "architecture": "multi-order context cache with longest-context backoff",
        "max_order": max(1, min(5, args.max_order)),
        "training_characters": len(text), "tables": tables,
    }
    target = OUT / "RC-Lilguy-v0.2.json"
    target.write_text(json.dumps(model, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Saved {target} | orders={model['max_order']} | {time.time()-start:.2f}s", flush=True)

if __name__ == "__main__":
    main()
