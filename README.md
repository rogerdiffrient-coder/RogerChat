# RogerChat

RogerChat is a small, experimental chat UI with several Lilguy model experiments. **Lilguy Bluejay v0.3** is the custom model trained from random initialization; it does not use pretrained model weights.

## Lilguy Bluejay v0.3

v0.3 is larger and is trained specifically to learn both dialogue and story-writing patterns.

- **Architecture:** decoder-only Transformer with 8 layers, 320 hidden dimensions, 8 attention heads, 2 key/value heads, RoPE, RMSNorm, SwiGLU, and grouped-query attention.
- **Tokenizer:** project-trained byte-level BPE tokenizer, with an 8,000-token target vocabulary.
- **Training data:** public UltraChat dialogue examples plus public TinyStories examples formatted as story-writing requests.
- **Default training run:** 3,000 dialogue examples, 3,000 story examples, 3,000 optimization steps, context length 384, batch size 4.
- **Inference:** CPU-friendly PyTorch generation; response length can be increased to 512 tokens in Settings.

This is still a small model trained from scratch, so coherent stories are a goal, not a guarantee. More training generally helps, but a short CPU-only run cannot match large pretrained language models. The model checkpoint is not included until the training workflow completes.

## Train v0.3 on GitHub Actions

1. Open the repository's **Actions** tab.
2. Select **Train Lilguy v0.3 — story-capable custom model**.
3. Click **Run workflow**.
4. Keep the default values to train on 3,000 dialogue examples and 3,000 story examples for 3,000 steps.
5. When the workflow finishes, it publishes a model artifact and attempts to commit the checkpoint, tokenizer, training log, and cached datasets.

The workflow uses a GitHub-hosted CPU runner. Runtime depends on runner load, dataset access, and GitHub usage limits. If the job times out, rerun with fewer steps or fewer examples.

## Run training locally

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_bluejay.py --steps 3000 --conversations 3000 --stories 3000 --threads 2
```

Use `--resume` only when continuing from a compatible v0.3 checkpoint. Older checkpoints from earlier model sizes are not compatible with the v0.3 architecture.

## Hosting the chat API

The frontend calls `/api/chat`. A static host such as GitHub Pages cannot execute the Python/PyTorch model endpoint; the frontend must be deployed with a compatible Python backend. The backend also needs the trained checkpoint at `checkpoints/RC-Lilguy-Bluejay.pt` and tokenizer at `checkpoints/bluejay_tokenizer.json`.

## Dataset sources

- [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) — dialogue examples.
- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) — short stories.

Review each dataset's card and license before redistributing derived data or model artifacts.


## Run locally from GitHub Pages (no Python API)

The **Lilguy Bluejay v0.3 · Local** browser runtime uses Transformers.js with the `onnx-community/SmolLM2-360M-Instruct-ONNX` model. The model runs on the visitor's device using WebGPU when available and WASM otherwise. The first use downloads the model weights into the browser cache (roughly a few hundred MB); later visits can reuse cached files when the browser retains them. The prompt is processed in the browser rather than sent to a RogerChat Python server. An internet connection is required for the initial library/model download. This browser runtime is not the same as loading the original custom-trained Bluejay checkpoint.
