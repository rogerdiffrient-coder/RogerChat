"""Serve both RC Lilguy experiments with Python's standard library."""
import json, random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import torch
from train import Lilguy, OUT

ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "index.html"
V1_CKPT = OUT / "RC-Lilguy-v0.1.pt"
V1_TOK = OUT / "tokenizer.json"
V2_PATH = OUT / "RC-Lilguy-v0.2.json"
v1_model = None
v1_chars = []
v1_stoi = {}
v2_data = None

def load_v1():
    global v1_model, v1_chars, v1_stoi
    if v1_model is not None:
        return
    if not V1_CKPT.exists() or not V1_TOK.exists():
        raise FileNotFoundError("v0.1 is not trained yet. Run the v0.1 workflow and add its checkpoint.")
    ck = torch.load(V1_CKPT, map_location="cpu", weights_only=False)
    v1_chars = json.loads(V1_TOK.read_text(encoding="utf-8"))["chars"]
    v1_stoi = {ch: i for i, ch in enumerate(v1_chars)}
    cfg = ck["config"]
    v1_model = Lilguy(len(v1_chars), ctx=cfg["context"], dim=cfg["dim"], layers=cfg["layers"], heads=cfg["heads"], dropout=0)
    v1_model.load_state_dict(ck["model"])
    v1_model.eval()

def load_v2():
    global v2_data
    if v2_data is None:
        if not V2_PATH.exists():
            raise FileNotFoundError("v0.2 is still training or has not been added yet.")
        v2_data = json.loads(V2_PATH.read_text(encoding="utf-8"))

def generate_v2(seed_text, n=120, temperature=0.8):
    """Sample from the longest matching context, backing off when unseen."""
    load_v2()
    tables = v2_data["tables"]
    order = int(v2_data.get("max_order", 5))
    chars = set(tables[0].get("", {}))
    output = seed_text
    for _ in range(n):
        chosen = None
        for size in range(min(order, len(output)), 0, -1):
            context = output[-size:]
            options = tables[size].get(context)
            if options:
                chosen = options
                break
        if not chosen:
            chosen = tables[0].get("", [])
        if not chosen:
            break
        # Temperature gently flattens the frequency distribution without
        # needing a neural forward pass.
        temp = max(0.15, float(temperature))
        weights = [max(1, int(count ** (1.0 / temp))) for _, count in chosen[:32]]
        pick = random.choices(chosen[:32], weights=weights, k=1)[0][0]
        output += pick
    return output[len(seed_text):]

class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/status":
            self._json({"models": {
                "RC-Lilguy-v0.1": V1_CKPT.exists() and V1_TOK.exists(),
                "RC-Lilguy-v0.2": V2_PATH.exists()
            }})
            return
        if self.path in ("/", "/index.html"):
            data = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(404)

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_error(404)
            return
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            message = str(payload.get("message", "")).strip()
            if not message:
                self._json({"reply": "Type a message first."}, 400)
                return
            model_id = payload.get("model_id", "RC-Lilguy-v0.1")
            tokens = min(220, max(1, int(payload.get("tokens", 120))))
            temperature = float(payload.get("temperature", 0.8))
            if model_id == "RC-Lilguy-v0.2":
                load_v2()
                # Only characters present in the learned corpus can condition generation.
                known = set("".join(v2_data["tables"][0].get("", {}).keys()))
                seed = "".join(ch for ch in ("User: " + message + "\nAssistant:") if ch in known)
                reply = generate_v2(seed, tokens, temperature)
            else:
                load_v1()
                seed = [v1_stoi[ch] for ch in ("User: " + message + "\nAssistant:") if ch in v1_stoi]
                if not seed:
                    seed = [0]
                with torch.no_grad():
                    out = v1_model.generate(torch.tensor([seed], dtype=torch.long), n=tokens, temp=temperature)
                reply = "".join(v1_chars[i] for i in out[0].tolist()[len(seed):]).strip()
            self._json({"reply": reply, "model_id": model_id})
        except FileNotFoundError as exc:
            self._json({"reply": str(exc)}, 503)
        except Exception as exc:
            self._json({"reply": "Generation error: " + str(exc)}, 500)

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))

if __name__ == "__main__":
    print("RogerChat running at http://127.0.0.1:8000")
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
