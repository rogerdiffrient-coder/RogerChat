"""Train Lilguy Bluejay from random initialization on conversational text.

This is a small research model, not a fine-tune of an existing model. Its
character-level tokenizer and Transformer weights are initialized from scratch.
"""
import argparse, json, time, urllib.parse, urllib.request
from pathlib import Path
import torch
from torch.nn import functional as F
from train import Lilguy, CharTokenizer, OUT

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL_ID = "RC-Lilguy-Bluejay"
MODEL_NAME = "Lilguy Bluejay"

def fetch_conversations(limit):
    DATA.mkdir(exist_ok=True)
    path = DATA / "bluejay_conversations.txt"
    if path.exists() and path.stat().st_size > 100_000:
        return path.read_text(encoding="utf-8")
    conversations, offset = [], 0
    print(f"Downloading up to {limit} public dialogue examples…", flush=True)
    while len(conversations) < limit:
        length = min(100, limit - len(conversations))
        params = urllib.parse.urlencode({
            "dataset": "HuggingFaceH4/ultrachat_200k",
            "config": "default", "split": "train_sft",
            "offset": offset, "length": length
        })
        url = "https://datasets-server.huggingface.co/rows?" + params
        req = urllib.request.Request(url, headers={"User-Agent": "RogerChat-Bluejay/0.1"})
        with urllib.request.urlopen(req, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
        rows = payload.get("rows", [])
        if not rows:
            break
        for item in rows:
            row = item.get("row", {})
            messages = row.get("messages", [])
            turns = []
            for msg in messages:
                role, content = msg.get("role", ""), msg.get("content", "")
                if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                    turns.append(("User" if role == "user" else "Bluejay") + ": " + content.strip())
            if len(turns) >= 2:
                conversations.append("\n".join(turns))
                if len(conversations) >= limit:
                    break
        offset += len(rows)
        print(f"Fetched {len(conversations)}/{limit}", flush=True)
        if len(rows) < length:
            break
    if not conversations:
        raise RuntimeError("No conversations downloaded; check the dataset viewer and network.")
    text = "\n\n<|conversation|>\n\n".join(conversations)
    path.write_text(text, encoding="utf-8")
    print(f"Saved {len(conversations)} conversations ({len(text):,} characters)", flush=True)
    return text

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--conversations", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--context", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=73)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(max(1, a.threads))
    text = fetch_conversations(a.conversations)
    tok = CharTokenizer(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    if len(ids) < a.context * 10:
        raise RuntimeError("Dialogue corpus is too small.")
    split = int(len(ids) * .98)
    tr, va = ids[:split], ids[split:]
    model = Lilguy(len(tok.chars), ctx=a.context).cpu()
    model.config.update({"model_id": MODEL_ID, "model_name": MODEL_NAME, "training": "from-scratch"})
    OUT.mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    checkpoint = OUT / "RC-Lilguy-Bluejay.pt"
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=.01)
    def batch(src):
        starts = torch.randint(0, len(src) - a.context - 1, (a.batch_size,))
        x = torch.stack([src[int(i):int(i)+a.context] for i in starts])
        y = torch.stack([src[int(i)+1:int(i)+a.context+1] for i in starts])
        return x, y
    @torch.no_grad()
    def evaluate(src):
        model.eval()
        vals = []
        for _ in range(4):
            x, y = batch(src)
            _, loss = model(x, y)
            vals.append(loss.item())
        model.train()
        return sum(vals) / len(vals)
    started = time.time()
    print(f"{MODEL_NAME} from scratch | {sum(p.numel() for p in model.parameters()):,} parameters | vocab={len(tok.chars)}", flush=True)
    for step in range(1, a.steps + 1):
        model.train()
        x, y = batch(tr)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % a.eval_every == 0 or step == a.steps:
            train_loss, val_loss = evaluate(tr), evaluate(va)
            tok.save(OUT / "bluejay_tokenizer.json")
            torch.save({"model": model.state_dict(), "config": model.config,
                        "step": step, "tokenizer": tok.chars}, checkpoint)
            with (ROOT / "logs" / "bluejay_training.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "train_loss": train_loss,
                    "val_loss": val_loss, "seconds": round(time.time()-started, 2)}) + "\n")
            print(f"step {step}/{a.steps} train={train_loss:.3f} val={val_loss:.3f}", flush=True)
    print(f"Saved custom checkpoint: {checkpoint}", flush=True)

if __name__ == "__main__":
    main()
