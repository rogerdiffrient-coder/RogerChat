"""Vercel-compatible inference endpoint for the from-scratch Lilguy Bluejay model."""
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler
import torch
from train import Lilguy

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints" / "RC-Lilguy-Bluejay.pt"
TOKENIZER = ROOT / "checkpoints" / "bluejay_tokenizer.json"
_model = None
_chars = None
_config = None

def load_model():
    global _model, _chars, _config
    if _model is not None:
        return _model, _chars, _config
    if not CHECKPOINT.exists() or not TOKENIZER.exists():
        raise RuntimeError("Bluejay has not been trained yet. Run the Train Lilguy Bluejay from scratch workflow in GitHub Actions.")
    ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    _config = ck["config"]
    _chars = json.loads(TOKENIZER.read_text(encoding="utf-8"))["chars"]
    _model = Lilguy(vocab=len(_chars), ctx=_config["context"], dim=_config["dim"],
                    layers=_config["layers"], heads=_config["heads"], dropout=_config["dropout"])
    _model.load_state_dict(ck["model"])
    _model.eval()
    return _model, _chars, _config

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
            if body.get("model_id") != "RC-Lilguy-Bluejay":
                return self._reply(400, {"reply": "This endpoint currently serves Lilguy Bluejay only."})
            prompt = str(body.get("message", "")).strip()[:3000]
            if not prompt:
                return self._reply(400, {"reply": "Type a message first."})
            model, chars, config = load_model()
            stoi = {ch: i for i, ch in enumerate(chars)}
            seed_text = "User: " + prompt + "\nBluejay:"
            seed = [stoi[ch] for ch in seed_text if ch in stoi]
            if not seed:
                return self._reply(500, {"reply": "The prompt contains no characters supported by Bluejay's tokenizer."})
            temp = max(0.2, min(1.2, float(body.get("temperature", 0.8))))
            count = max(20, min(220, int(body.get("tokens", 120))))
            with torch.no_grad():
                output = model.generate(torch.tensor([seed], dtype=torch.long), n=count, temp=temp)[0].tolist()
            decoded = "".join(chars[int(i)] for i in output[len(seed):])
            for marker in ("\nUser:", "\n<|conversation|>"):
                if marker in decoded:
                    decoded = decoded.split(marker, 1)[0]
            reply = decoded.strip()
            if not reply:
                reply = "I couldn't form a reply yet. Bluejay is still an experimental small model."
            return self._reply(200, {"reply": reply})
        except Exception as exc:
            return self._reply(500, {"reply": str(exc)})

    def do_GET(self):
        return self._reply(200, {"model": "Lilguy Bluejay", "status": "custom from-scratch Transformer"})

# Some Python hosting adapters look for a lowercase handler export.
handler = Handler
