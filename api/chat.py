"""Inference endpoint for the trained from-scratch Lilguy Bluejay model."""
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler

import torch
from tokenizers import Tokenizer
from bluejay_model import BluejayLM

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints" / "RC-Lilguy-Bluejay.pt"
TOKENIZER_PATH = ROOT / "checkpoints" / "bluejay_tokenizer.json"
_model = None
_tokenizer = None
_config = None


def load_model():
    global _model, _tokenizer, _config
    if _model is not None:
        return _model, _tokenizer, _config
    if not CHECKPOINT.exists() or not TOKENIZER_PATH.exists():
        raise RuntimeError("Bluejay weights are not trained yet. Run the Train Lilguy Bluejay workflow.")
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    _config = saved["config"]
    _tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    _model = BluejayLM(
        vocab_size=_config["vocab_size"], context=_config["context"],
        dim=_config["dim"], layers=_config["layers"], heads=_config["heads"],
        kv_heads=_config["kv_heads"], dropout=_config.get("dropout", 0.0)
    )
    _model.load_state_dict(saved["model"])
    _model.eval()
    return _model, _tokenizer, _config


class Handler(BaseHTTPRequestHandler):
    def _reply(self, status, payload):
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        try:
            length = min(int(self.headers.get("Content-Length", "0")), 100_000)
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if body.get("model_id") != "RC-Lilguy-v0.3":
                return self._reply(400, {"reply": "This endpoint currently serves Lilguy Bluejay only."})
            prompt = str(body.get("message", "")).strip()[:3000]
            if not prompt:
                return self._reply(400, {"reply": "Type a message first."})
            model, tokenizer, config = load_model()
            seed_text = "<|user|> " + prompt + "\n<|assistant|>"
            encoded = tokenizer.encode(seed_text).ids
            if not encoded:
                return self._reply(400, {"reply": "Bluejay could not tokenize that message."})
            temperature = max(0.15, min(1.2, float(body.get("temperature", 0.75))))
            count = max(16, min(512, int(body.get("tokens", 320))))
            with torch.no_grad():
                output = model.generate(
                    torch.tensor([encoded], dtype=torch.long),
                    max_new_tokens=count, temperature=temperature, top_k=40,
                    eos_id=config.get("eos_token_id")
                )[0].tolist()
            reply = tokenizer.decode(output[len(encoded):], skip_special_tokens=True)
            for marker in ("<|user|>", "<|assistant|>", "<|end|>"):
                if marker in reply:
                    reply = reply.split(marker, 1)[0]
            reply = reply.strip()
            if not reply:
                reply = "I don't have a useful answer yet—I'm still learning."
            return self._reply(200, {"reply": reply})
        except Exception as exc:
            return self._reply(500, {"reply": str(exc)})

    def do_GET(self):
        return self._reply(200, {"model": "Lilguy Bluejay v0.3", "status": "larger from-scratch Transformer; dialogue + story training; RoPE, RMSNorm, SwiGLU, GQA, and BPE"})


handler = Handler
