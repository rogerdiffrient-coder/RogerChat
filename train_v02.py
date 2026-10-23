"""Train RC Lilguy v0.2: a CPU-first multi-order context-cache language model.

This is an experimental alternative to v0.1's neural Transformer. It learns
next-character distributions from 1- through 5-character contexts in one pass.
It is fast and compact, but it is not a semantic reasoning model.
"""
import argparse, json, random, time
from collections import Counter, defaultdict
import urllib.parse, urllib.request

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "checkpoints"

def fetch_stories(max_stories=800):
    DATA.mkdir(exist_ok=True)
    path = DATA / "tinystories.txt"
    if path.exists() and path.stat().st_size > 100_000: return path.read_text(encoding="utf-8")
    stories, offset = [], 0
    while len(stories) < max_stories:
        length = min(100, max_stories - len(stories))
        params = urllib.parse.urlencode({"dataset":"roneneldan/TinyStories","config":"default","split":"train","offset":offset,"length":length})
        host = "https://datasets-server." + "huggingface.co/rows?"
        request = urllib.request.Request(host + params, headers={"User-Agent":"RogerChat-Research/0.2"})
        with urllib.request.urlopen(request, timeout=30) as response: payload = json.loads(response.read().decode("utf-8"))
        rows = payload.get("rows", [])
        if not rows: break
        for row in rows:
            value = row.get("row", {}).get("text", "")
            if isinstance(value, str) and value.strip(): stories.append(value.strip())
        offset += len(rows)
        print("Fetched %d/%d stories" % (len(stories), max_stories), flush=True)
        if len(rows) < length: break
    if not stories: raise RuntimeError("No stories downloaded.")
    text = "\n\n".join(stories)
    path.write_text(text, encoding="utf-8")
    return text
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
