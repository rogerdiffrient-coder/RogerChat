import argparse, json
from pathlib import Path
import torch
from train import Lilguy, CharTokenizer, OUT

p=argparse.ArgumentParser()
p.add_argument("--prompt",default="Once upon a time")
p.add_argument("--tokens",type=int,default=160)
p.add_argument("--temperature",type=float,default=0.8)
a=p.parse_args()
ckpt_path=OUT/"RC-Lilguy-v0.1.pt"
tok_path=OUT/"tokenizer.json"
if not ckpt_path.exists() or not tok_path.exists():
    raise SystemExit("No trained checkpoint yet. Run python train.py first.")
ck=torch.load(ckpt_path,map_location="cpu",weights_only=False)
chars=json.loads(tok_path.read_text(encoding="utf-8"))["chars"]
tok=CharTokenizer.__new__(CharTokenizer); tok.chars=chars; tok.stoi={c:i for i,c in enumerate(chars)}; tok.itos=chars
model=Lilguy(**{"vocab":len(chars),"ctx":ck["config"]["context"],"dim":ck["config"]["dim"],"layers":ck["config"]["layers"],"heads":ck["config"]["heads"],"dropout":ck["config"]["dropout"]})
model.load_state_dict(ck["model"]); model.eval()
seed=[tok.stoi[c] for c in a.prompt if c in tok.stoi] or [0]
with torch.no_grad(): out=model.generate(torch.tensor([seed]),a.tokens,a.temperature)[0].tolist()
print(tok.decode(out))
