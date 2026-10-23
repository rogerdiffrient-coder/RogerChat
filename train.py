"""Train RC Lilguy v0.1 from random initialization on a CPU-first setup."""
import argparse, json, math, os, random, time, urllib.parse, urllib.request
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "checkpoints"
MODEL_ID = "RC-Lilguy-v0.1"
MODEL_NAME = "RC Lilguy v0.1"

def fetch_stories(max_stories=1200):
    DATA.mkdir(exist_ok=True)
    path = DATA / "tinystories.txt"
    if path.exists() and path.stat().st_size > 100_000:
        return path.read_text(encoding="utf-8")
    print(f"Downloading up to {max_stories} TinyStories examples via the Hugging Face dataset viewer…", flush=True)
    stories, offset = [], 0
    while len(stories) < max_stories:
        length = min(100, max_stories - len(stories))
        params = urllib.parse.urlencode({"dataset":"roneneldan/TinyStories","config":"default","split":"train","offset":offset,"length":length})
        url = "https://datasets-server.huggingface.co/rows?" + params
        req = urllib.request.Request(url, headers={"User-Agent":"RogerChat-Research/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            if stories:
                print(f"Dataset fetch stopped after {len(stories)} examples: {exc}", flush=True)
                break
            raise RuntimeError("Could not fetch TinyStories. Check internet access and dataset viewer availability.") from exc
        rows = payload.get("rows", [])
        if not rows: break
        for row in rows:
            item = row.get("row", {})
            text = item.get("text", "")
            if isinstance(text, str) and text.strip():
                stories.append(text.strip())
        offset += len(rows)
        print(f"Fetched {len(stories)}/{max_stories} stories", flush=True)
        if len(rows) < length: break
    if not stories:
        raise RuntimeError("No stories were downloaded.")
    text = "\n\n".join(stories)
    path.write_text(text, encoding="utf-8")
    print(f"Dataset saved: {len(text):,} characters", flush=True)
    return text

class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {ch:i for i,ch in enumerate(self.chars)}
        self.itos = self.chars
    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]
    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids if 0 <= int(i) < len(self.itos))
    def save(self, path):
        path.write_text(json.dumps({"chars":self.chars}, ensure_ascii=False), encoding="utf-8")

class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads, self.hdim = heads, dim // heads
        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)
        self.drop = dropout
    def forward(self, x):
        b,t,c=x.shape
        q,k,v=self.qkv(x).chunk(3,dim=-1)
        q=q.view(b,t,self.heads,self.hdim).transpose(1,2)
        k=k.view(b,t,self.heads,self.hdim).transpose(1,2)
        v=v.view(b,t,self.heads,self.hdim).transpose(1,2)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=self.drop if self.training else 0.0)
        return self.proj(y.transpose(1,2).contiguous().view(b,t,c))

class Block(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.n1=nn.LayerNorm(dim); self.attn=Attention(dim,heads,dropout)
        self.n2=nn.LayerNorm(dim)
        self.mlp=nn.Sequential(nn.Linear(dim,4*dim),nn.GELU(),nn.Linear(4*dim,dim),nn.Dropout(dropout))
    def forward(self,x): return x+self.attn(self.n1(x)) if False else self._forward(x)
    def _forward(self,x):
        x=x+self.attn(self.n1(x))
        return x+self.mlp(self.n2(x))

class Lilguy(nn.Module):
    def __init__(self,vocab,ctx=96,dim=96,layers=2,heads=4,dropout=0.0):
        super().__init__()
        self.config={"model_id":MODEL_ID,"model_name":MODEL_NAME,"vocab_size":vocab,"context":ctx,"dim":dim,"layers":layers,"heads":heads,"dropout":dropout,"architecture":"decoder-only Transformer"}
        self.tok=nn.Embedding(vocab,dim); self.pos=nn.Embedding(ctx,dim)
        self.blocks=nn.Sequential(*[Block(dim,heads,dropout) for _ in range(layers)])
        self.norm=nn.LayerNorm(dim); self.head=nn.Linear(dim,vocab,bias=False)
        self.head.weight=self.tok.weight
        self.apply(self.init_weights)
    @staticmethod
    def init_weights(m):
        if isinstance(m,nn.Linear):
            nn.init.normal_(m.weight,0,0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,0,0.02)
    def forward(self,idx,targets=None):
        _,t=idx.shape
        x=self.tok(idx)+self.pos(torch.arange(t,device=idx.device))
        logits=self.head(self.norm(self.blocks(x)))
        loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1)) if targets is not None else None
        return logits,loss
    @torch.no_grad()
    def generate(self,idx,n=160,temp=0.8,top_k=24):
        self.eval()
        for _ in range(n):
            logits,_=self(idx[:,-self.config["context"]:])
            logits=logits[:,-1,:]/max(temp,0.05)
            if top_k:
                k=min(top_k,logits.size(-1)); vals,_=torch.topk(logits,k)
                logits[logits<vals[:,-1,None]]=-float("inf")
            idx=torch.cat((idx,torch.multinomial(F.softmax(logits,dim=-1),1)),dim=1)
        return idx

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--steps",type=int,default=500); p.add_argument("--stories",type=int,default=1200)
    p.add_argument("--batch-size",type=int,default=32); p.add_argument("--context",type=int,default=96)
    p.add_argument("--eval-every",type=int,default=100); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--threads",type=int,default=2); p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(max(1,args.threads))
    device=torch.device("cpu")
    text=fetch_stories(args.stories)
    tok=CharTokenizer(text); ids=torch.tensor(tok.encode(text),dtype=torch.long)
    if len(ids)<args.context*10: raise RuntimeError("Dataset is too small for this context length.")
    split=int(len(ids)*0.98); tr,va=ids[:split],ids[split:]
    if len(va)<args.context+2: va=tr[-max(2000,args.context*10):]
    model=Lilguy(len(tok.chars),ctx=args.context).to(device)
    OUT.mkdir(exist_ok=True); (ROOT/"logs").mkdir(exist_ok=True)
    checkpoint=OUT/"RC-Lilguy-v0.1.pt"
    start=0
    if args.resume and checkpoint.exists():
        ck=torch.load(checkpoint,map_location="cpu",weights_only=False)
        model.load_state_dict(ck["model"]); start=ck.get("step",0)
        print(f"Resuming from step {start}",flush=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=0.01)
    params=sum(x.numel() for x in model.parameters())
    print(f"{MODEL_NAME} | {params:,} parameters | CPU | vocab={len(tok.chars)} | data={len(ids):,} chars",flush=True)
    def batch(src):
        starts=torch.randint(0,len(src)-args.context-1,(args.batch_size,))
        x=torch.stack([src[int(i):int(i)+args.context] for i in starts])
        y=torch.stack([src[int(i)+1:int(i)+args.context+1] for i in starts])
        return x,y
    @torch.no_grad()
    def eval_loss(src,n=4):
        model.eval(); vals=[]
        for _ in range(n):
            x,y=batch(src); _,loss=model(x,y); vals.append(loss.item())
        model.train(); return sum(vals)/len(vals)
    t0=time.time()
    for step in range(start+1,start+args.steps+1):
        model.train(); x,y=batch(tr); _,loss=model(x,y)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
        if step==1 or step%args.eval_every==0 or step==start+args.steps:
            train_loss=eval_loss(tr); val_loss=eval_loss(va)
            elapsed=time.time()-t0
            print(f"step {step:5d} | train_loss {train_loss:.3f} | val_loss {val_loss:.3f} | {elapsed:.1f}s",flush=True)
            tok.save(OUT/"tokenizer.json")
            torch.save({"model":model.state_dict(),"config":model.config,"step":step,"tokenizer":tok.chars},checkpoint)
            with (ROOT/"logs/training.jsonl").open("a",encoding="utf-8") as f:
                f.write(json.dumps({"step":step,"train_loss":train_loss,"val_loss":val_loss,"seconds":round(elapsed,2)})+"\n")
    prompt="Once upon a time"
    seed=[tok.stoi[c] for c in prompt if c in tok.stoi] or [0]
    sample=model.generate(torch.tensor([seed]),n=240)[0].tolist()
    print("\n--- SAMPLE ---\n"+tok.decode(sample)+"\n--- END SAMPLE ---",flush=True)
    (OUT/"sample.txt").write_text(tok.decode(sample),encoding="utf-8")
    print(f"Checkpoint saved to {checkpoint}",flush=True)

if __name__=="__main__": main()
