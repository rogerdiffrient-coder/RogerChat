"""Serve the trained RC Lilguy model with Python's standard library only."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import torch
from train import Lilguy, OUT

ROOT=Path(__file__).resolve().parent
PAGE=ROOT/"index.html"
ckpt_path=OUT/"RC-Lilguy-v0.1.pt"
tok_path=OUT/"tokenizer.json"
model=None
chars=[]
stoi={}

def load_model():
    global model,chars,stoi
    if model is not None: return
    if not ckpt_path.exists() or not tok_path.exists():
        raise FileNotFoundError("Train RC Lilguy first, then place its checkpoint and tokenizer in checkpoints/.")
    ck=torch.load(ckpt_path,map_location="cpu",weights_only=False)
    chars=json.loads(tok_path.read_text(encoding="utf-8"))["chars"]
    stoi={c:i for i,c in enumerate(chars)}
    c=ck["config"]
    model=Lilguy(len(chars),ctx=c["context"],dim=c["dim"],layers=c["layers"],heads=c["heads"],dropout=0)
    model.load_state_dict(ck["model"]); model.eval()

class Handler(BaseHTTPRequestHandler):
    def _json(self,obj,status=200):
        data=json.dumps(obj,ensure_ascii=False).encode("utf-8")
        self.send_response(status); self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Content-Length",str(len(data))); self.send_header("Access-Control-Allow-Origin","*"); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        if self.path=="/api/status":
            self._json({"model_id":"RC-Lilguy-v0.1","model_name":"RC Lilguy v0.1","trained":ckpt_path.exists() and tok_path.exists()}); return
        if self.path in ("/","/index.html"):
            data=PAGE.read_bytes(); self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8"); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data); return
        self.send_error(404)
    def do_POST(self):
        if self.path!="/api/chat": self.send_error(404); return
        try:
            payload=json.loads(self.rfile.read(int(self.headers.get("Content-Length","0"))))
            message=str(payload.get("message","")).strip()
            if not message: self._json({"reply":"Type a message first."},400); return
            load_model()
            seed=[stoi[c] for c in ("User: "+message+"\nAssistant:") if c in stoi]
            if not seed: seed=[0]
            with torch.no_grad():
                out=model.generate(torch.tensor([seed],dtype=torch.long),n=min(220,max(1,int(payload.get("tokens",120)))),temp=float(payload.get("temperature",0.8)))
            reply="".join(chars[i] for i in out[0].tolist()[len(seed):])
            self._json({"reply":reply.strip(),"model_id":"RC-Lilguy-v0.1"})
        except FileNotFoundError as exc: self._json({"reply":str(exc)},503)
        except Exception as exc: self._json({"reply":"Generation error: "+str(exc)},500)
    def log_message(self,fmt,*args): print("%s - %s"%(self.address_string(),fmt%args))

if __name__=="__main__":
    print("RogerChat running at http://127.0.0.1:8000")
    ThreadingHTTPServer(("127.0.0.1",8000),Handler).serve_forever()
