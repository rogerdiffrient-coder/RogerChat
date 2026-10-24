"""Train Lilguy Bluejay from random initialization with a modern small-LM stack.

No pretrained weights are loaded. The BPE tokenizer and every model parameter are
trained/initialized for this project. This is a research-scale model, not a
claim of ChatGPT-level capability.
"""
import argparse
import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from bluejay_model import BluejayLM

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "checkpoints"
SPECIAL = ["<|pad|>", "<|user|>", "<|assistant|>", "<|end|>", "<|turn|>"]


def fetch_conversations(limit):
    DATA.mkdir(exist_ok=True)
    path = DATA / "bluejay_conversations.jsonl"
    conversations = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                if item.get("text"):
                    conversations.append(item["text"])
            except json.JSONDecodeError:
                continue
        if len(conversations) >= min(limit, 100):
            print(f"Using cached {len(conversations)} conversations", flush=True)
            return conversations[:limit]

    print(f"Downloading up to {limit} dialogue examples from UltraChat 200k…", flush=True)
    offset = 0
    while len(conversations) < limit:
        length = min(100, limit - len(conversations))
        params = urllib.parse.urlencode({
            "dataset": "HuggingFaceH4/ultrachat_200k",
            "config": "default", "split": "train_sft",
            "offset": offset, "length": length
        })
        url = "https://datasets-server.huggingface.co/rows?" + params
        req = urllib.request.Request(url, headers={"User-Agent": "RogerChat-Bluejay/0.3"})
        with urllib.request.urlopen(req, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        rows = payload.get("rows", [])
        if not rows:
            break
        for item in rows:
            messages = item.get("row", {}).get("messages", [])
            parts = []
            for message in messages:
                role = message.get("role")
                content = message.get("content", "")
                if role == "user" and isinstance(content, str) and content.strip():
                    parts.append("<|user|> " + content.strip())
                elif role == "assistant" and isinstance(content, str) and content.strip():
                    parts.append("<|assistant|> " + content.strip() + " <|end|>")
            if any(p.startswith("<|user|>") for p in parts) and any(p.startswith("<|assistant|>") for p in parts):
                conversations.append("\n".join(parts))
                if len(conversations) >= limit:
                    break
        offset += len(rows)
        print(f"Fetched {len(conversations)}/{limit}", flush=True)
        if len(rows) < length:
            break
    if not conversations:
        raise RuntimeError("No conversations downloaded from the dataset viewer.")
    path.write_text("".join(json.dumps({"text": c}, ensure_ascii=False) + "\n" for c in conversations), encoding="utf-8")
    return conversations



def fetch_stories(limit):
    """Fetch real short-story examples and convert them into story-writing instruction pairs."""
    DATA.mkdir(exist_ok=True)
    path = DATA / "bluejay_stories.jsonl"
    stories = []
    story_prompts = [
        "Write a short, original story.",
        "Write me a story.",
        "Tell me a story.",
        "Create an original story with a beginning, middle, and ending.",
        "Can you write a story for me?",
        "Make up a fun little story."
    ]
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                if item.get("text"):
                    stories.append(item["text"])
            except json.JSONDecodeError:
                continue
        if len(stories) >= min(limit, 100):
            print(f"Using cached {len(stories)} stories", flush=True)
            return stories[:limit]

    print(f"Downloading up to {limit} story examples from TinyStories…", flush=True)
    offset = 0
    while len(stories) < limit:
        length = min(100, limit - len(stories))
        params = urllib.parse.urlencode({
            "dataset": "roneneldan/TinyStories",
            "config": "default", "split": "train",
            "offset": offset, "length": length
        })
        url = "https://datasets-server.huggingface.co/rows?" + params
        req = urllib.request.Request(url, headers={"User-Agent": "RogerChat-Bluejay/0.3"})
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            print(f"Story dataset fetch stopped at {len(stories)} examples: {exc}", flush=True)
            break
        rows = payload.get("rows", [])
        if not rows:
            break
        for item in rows:
            row = item.get("row", {})
            story = row.get("text", "")
            if isinstance(story, str) and len(story.strip()) >= 80:
                story = story.strip()
                prompt = story_prompts[len(stories) % len(story_prompts)]
                stories.append("<|user|> " + prompt + "\n<|assistant|> " + story + " <|end|>")
                if len(stories) >= limit:
                    break
        offset += len(rows)
        print(f"Fetched {len(stories)}/{limit} stories", flush=True)
        if len(rows) < length:
            break
    if not stories:
        raise RuntimeError("No stories downloaded. Check access to the TinyStories dataset viewer.")
    path.write_text("".join(json.dumps({"text": story}, ensure_ascii=False) + "\n" for story in stories), encoding="utf-8")
    return stories


def make_tokenizer(texts, vocab_size, path):
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2,
        special_tokens=["<|unk|>"] + SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tokenizer.train_from_iterator(texts, trainer=trainer, length=len(texts))
    tokenizer.save(str(path))
    return tokenizer


def encode_dataset(tokenizer, texts):
    eos = tokenizer.token_to_id("<|end|>")
    ids = []
    for text in texts:
        ids.extend(tokenizer.encode(text).ids)
        if not ids or ids[-1] != eos:
            ids.append(eos)
    return torch.tensor(ids, dtype=torch.long)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--conversations", type=int, default=3000)
    p.add_argument("--stories", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--context", type=int, default=384)
    p.add_argument("--vocab-size", type=int, default=8000)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=73)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(max(1, a.threads))
    OUT.mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    conversations = fetch_conversations(a.conversations)
    stories = fetch_stories(a.stories)
    texts = conversations + stories
    random.shuffle(texts)
    split_at = max(1, int(len(texts) * .98))
    train_texts, val_texts = texts[:split_at], texts[split_at:]
    if not val_texts:
        val_texts = train_texts[-1:]

    tokenizer_path = OUT / "bluejay_tokenizer.json"
    tokenizer = make_tokenizer(train_texts, a.vocab_size, tokenizer_path)
    train_ids = encode_dataset(tokenizer, train_texts)
    val_ids = encode_dataset(tokenizer, val_texts)
    if len(train_ids) < a.context + 2:
        raise RuntimeError("Not enough tokenized training data for the selected context.")

    model = BluejayLM(tokenizer.get_vocab_size(), context=a.context, dim=320, layers=8, heads=8, kv_heads=2).cpu()
    checkpoint = OUT / "RC-Lilguy-Bluejay.pt"
    start_step = 0
    if a.resume and checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        start_step = int(saved.get("step", 0))
        print(f"Resuming at step {start_step}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Bluejay v0.3 from scratch | {total_params:,} parameters | RoPE + RMSNorm + SwiGLU + GQA | BPE vocab={tokenizer.get_vocab_size()}", flush=True)

    def batch(src):
        starts = torch.randint(0, len(src) - a.context - 1, (a.batch_size,))
        x = torch.stack([src[int(i):int(i) + a.context] for i in starts])
        y = torch.stack([src[int(i) + 1:int(i) + a.context + 1] for i in starts])
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
    log_path = ROOT / "logs" / "bluejay_training.jsonl"
    for step in range(start_step + 1, start_step + a.steps + 1):
        model.train()
        x, y = batch(train_ids)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % a.eval_every == 0 or step == start_step + a.steps:
            train_loss = evaluate(train_ids)
            val_loss = evaluate(val_ids)
            config = dict(model.config)
            config.update({"step": step, "tokenizer_file": "bluejay_tokenizer.json",
                           "special_tokens": SPECIAL, "training_examples": len(texts), "story_examples": len(stories), "eos_token_id": tokenizer.token_to_id("<|end|>"),
                           "pad_token_id": tokenizer.token_to_id("<|pad|>")})
            torch.save({"model": model.state_dict(), "config": config, "step": step}, checkpoint)
            record = {"step": step, "train_loss": train_loss, "val_loss": val_loss,
                      "seconds": round(time.time() - started, 2), "parameters": total_params}
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            print(f"step {step}/{start_step + a.steps} train={train_loss:.3f} val={val_loss:.3f}", flush=True)

    print(f"Saved trained custom weights to {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
