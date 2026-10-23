# RogerChat

**Models:** RC Lilguy v0.1 (Transformer), v0.2 (Context Cache), and v0.3 (Qwen2.5-0.5B-Instruct via WebLLM/WebGPU).  
**Goal:** Keep the original experiments while adding a genuinely pretrained instruction model for more coherent chat.

## Speed-first design
- Character tokenizer built from training text; no tokenizer training step.
- 2 Transformer blocks, 4 attention heads, 96-dimensional hidden state.
- 96-character context and tied token/output embeddings.
- Small batches, AdamW, periodic checkpointing.
- No pretrained weights and no GPU requirement.

This architecture favors fast experiments over broad knowledge. Character models need more steps to learn fluent text than subword models, so early outputs may be incoherent. The goal is a quick, reproducible training pipeline before scaling.

## Train on GitHub Actions CPU
1. Open the repository's **Actions** tab.
2. Select **Train RC Lilguy v0.1**.
3. Click **Run workflow**.
4. Start with **300 steps** and **800 stories**.
5. Download the `RC-Lilguy-v0.1-checkpoint` artifact from the completed workflow run.

The workflow uses a standard GitHub-hosted Ubuntu CPU runner. Free usage depends on account limits and GitHub's current policies. Runtime varies with runner load, package installation, and dataset availability.

## Train locally
Python 3.11 recommended:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --steps 300 --stories 800
python generate.py --prompt "Once upon a time"
```
Continue from the latest checkpoint with `--resume`; `--steps` means additional steps when resuming.

## Dataset
The training script fetches a bounded number of examples from the Hugging Face dataset viewer for `roneneldan/TinyStories`. Review its dataset card and terms: https://huggingface.co/datasets/roneneldan/TinyStories

## Honest limitations
This is a language-model experiment, not a capable assistant yet. It predicts the next character and has no instruction tuning, conversation fine-tuning, retrieval, or factuality guarantees. Training loss alone does not prove that outputs are useful. Model files are uploaded as short-lived workflow artifacts; the workflow does not deploy a chatbot server.


## v0.3 — pretrained browser model
Choose **v0.3 · Qwen Instruct** in the model picker. The first message loads the WebLLM runtime and downloads the quantized Qwen2.5-0.5B-Instruct model into the browser; this can be several hundred megabytes and needs WebGPU plus a modern supported browser. Later messages reuse the loaded model. Model weights run on the user's device; the initial runtime/model download requires internet. This is a real pretrained instruction model, unlike v0.2's character-context cache, but at 0.5B parameters it is still a small model and will not match ChatGPT-class systems.
