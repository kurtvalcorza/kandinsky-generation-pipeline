# Weight provenance, the three pinned snapshots and DIMER hosting

This repository pins **three** Hugging Face snapshots, each with its own `dimer-base-manifest.json` (format `dimer_hf_snapshot` v1: byte size and SHA-256 per file, the immutable revision, the licence) and each staged and verified separately by `src/kandinsky_generation_pipeline/pipeline.py`. Every weight file is safetensors; nothing is unpickled, and the model classes come from `diffusers`, `transformers` and `peft` on PyPI — no Hub-hosted code is executed.

## 1. The decoder — Kandinsky 2.2 Decoder

- Upstream: `kandinsky-community/kandinsky-2-2-decoder`
- Immutable revision: `9ae140d347fed8ce6e8bb3005dcc1f48543bb8e3`
- Upstream weight license: **Apache-2.0** (`license: apache-2.0` in the pinned README front matter). The licence permits use, redistribution and commercial use.
- Local layout: `weights/kandinsky-2-2-decoder/` holds the 7 manifest entries — upstream `README.md`, `model_index.json`, `scheduler/scheduler_config.json`, `movq/config.json`, `movq/diffusion_pytorch_model.safetensors` (271,380,364 bytes), `unet/config.json`, and `unet/diffusion_pytorch_model.safetensors` (5,012,319,952 bytes); 5,283,700,183 bytes in total. `verify_snapshot()` checks every entry by byte size and SHA-256 and refuses on the first mismatch; `stage_missing_files(allow_download=True)` fetches only absent entries, only at the pinned revision.
- Architecture: `UNet2DConditionModel` with 1,224,965,124 parameters (1.25 B), conditioned on CLIP image embeddings (cross-attention to 768 / 1,280 dims) and timesteps; MoVQ VQModel (67 M parameters). Loaded in float16 on CUDA. Note: upstream repository also ships duplicate `.bin` files; only `.safetensors` files are tracked and staged.

## 2. The prior — Kandinsky 2.2 Prior (Shared)

- Upstream: `kandinsky-community/kandinsky-2-2-prior`
- Immutable revision: `9fc51ad5732afc5d031724219d22e6c42179c5a8`
- Upstream license: **Apache-2.0** in the pinned README front matter.
- Local layout: `weights/kandinsky-2-2-prior/` holds the 14 manifest entries (10,574,964,619 bytes): `README.md`, `model_index.json`, `prior/config.json`, `prior/diffusion_pytorch_model.safetensors` (4,103,117,144 bytes, `PriorTransformer`), `image_encoder/config.json`, `image_encoder/model.safetensors` (3,689,837,458 bytes, CLIP ViT-G/14), `image_processor/preprocessor_config.json`, `text_encoder/config.json`, `text_encoder/model.safetensors` (2,781,775,178 bytes, CLIP text encoder), `tokenizer/{tokenizer_config.json, vocab.json, merges.txt, special_tokens_map.json}`, and `scheduler/scheduler_config.json`.
- The prior is shared across all three Kandinsky 2.2 pipeline profiles. It is loaded by `encode_prompts` and released by `release_prior`, because keeping both prior and decoder in memory during training would reduce available VRAM.

## 3. The scorer — CLIP ViT-B/32 (evaluation only)

- Upstream: `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`, revision `1a25a446712ba5ee05982a381eed697ef9b435cf`, MIT licence.
- Local layout: `weights/clip-vit-b-32-laion2b/`, 9 manifest entries (608,782,299 bytes): `model.safetensors` (605,157,884 bytes) plus config, tokenizer and preprocessor files. Loaded as `transformers.CLIPModel` in float32, frozen; used by `metrics.ClipScorer` only — never part of generation, never trained, and its scores are not a human judgement of image quality.

## Fidelity

No upstream regression fixture is published for the UNet. The evidence is the strict load of base tensors into `UNet2DConditionModel` from the pinned `config.json`, parameter count assertion, and generated sample validation: on the feasibility run, the frozen model generates 512² images in 14.70 s at 20 steps with peak allocated VRAM of 3,224.4 MB without requiring CPU offload.

## Runtime facts

- Precision: decoder UNet float16 on CUDA (float32 on CPU, unsupported for the tutorial), LoRA parameters float32 with float16 autocast, MoVQ float16, prior pipeline float16 on CUDA.
- Generation: 20 steps at guidance 4.0 by default (`MAX_STEPS = 100`, `MAX_GUIDANCE = 20.0`), CPU generator for reproducibility.
- Evaluation: held-out denoising MSE on MoVQ-encoded latents at fixed timesteps with seeded noise.
- Adaptation: LoRA rank 8 / alpha 8 on `to_q`, `to_k`, `to_v`, `to_out.0` of attention blocks — 176 tensors, 1,646,592 parameters — injected via `peft`; AdamW, gradient clipping norm 1.0, uniform timesteps, validation-loss epoch selection.
- `diffusers`, `transformers`, `peft`, `accelerate` are runtime pins beyond torch; the package imports all of them lazily so manifest verification and input validation run first (fleet RTM-001).

## DIMER hosting

- **Licence gate:** Apache-2.0 permissive licence for both decoder and prior upstreams.
- **Size gate (>9 GB waiver):** The served set is approximately 15.9 GB (5.28 GB decoder + 10.57 GB prior). This exceeds the fleet's 9 GB convenience gate, but following the Toto / PixArt-Σ precedent, the gate is waived and full weights are published.
- **Safetensors only:** The upstream repositories ship duplicate `.bin` weights alongside safetensors. Only `.safetensors` files are staged, avoiding downloading duplicate weights.
