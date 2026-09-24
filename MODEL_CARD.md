---
license: apache-2.0
model_card_spec: "1.2"
pipeline_tag: text-to-image
task: "Text-to-Image Generation (Kandinsky 2.2 diffusion UNet + MoVQ)"
base_model: kandinsky-community/kandinsky-2-2-decoder
date_published: "2023-07-12"
date_published_source: "Hugging Face Hub commit `9ae140d347fed8ce6e8bb3005dcc1f48543bb8e3` on 2023-07-12; safetensors weights published with Apache-2.0 license"
---

# Kandinsky 2.2 Decoder — Text-to-Image Generation (Captioned Photographs & Bounded LoRA Fine-Tuning)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-kandinsky--community%2Fkandinsky--2--2--decoder-ffcc4d?style=flat)](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-ai--forever%2Fkandinsky-181717?style=flat&logo=github&logoColor=white)](https://github.com/ai-forever/kandinsky-2)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder/blob/main/README.md)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream Apache-2.0 license; the accompanying code and notebook are Apache-2.0. Nothing here is validated for production, and no benchmark result is claimed.

> [!IMPORTANT]
> **Three pinned snapshots make one generator, the prior pipeline can be released before training, and generation has no ground truth.** The decoder checkpoint ships the 5.01 GB UNet and 0.27 GB MoVQ; the prior repository provides the CLIP text and image encoders and prior diffusion transformer, and a CLIP ViT-B/32 is pinned for evaluation only. `src/kandinsky_generation_pipeline/pipeline.py` stages and digest-verifies all three, encodes prompts once and releases the prior pipeline before training, and scores the model with a held-out denoising loss and CLIP scores against the real-photo ceiling.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end:

- **End-to-End Text-to-Image Fine-Tuning Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/kandinsky-generation-pipeline/blob/main/tutorials/kandinsky_generation_colab.ipynb) [`kandinsky_generation_colab.ipynb`](https://github.com/kurtvalcorza/kandinsky-generation-pipeline/blob/main/tutorials/kandinsky_generation_colab.ipynb)  
  *End-to-end LoRA fine-tuning of Kandinsky 2.2 with pinned snapshots: 60 digest-pinned CC0 iNaturalist bird photographs with a seeded stratified split, structural validation with refusal probes, prompt encoding with the prior pipeline released before training, held-out denoising loss and CLIP-scored generations against the real-photo ceiling for the frozen model, bounded LoRA fine-tuning of the UNet attention projections, a paired comparison on identical held-out inputs, a new prompt rendered, and safetensors adapter export with verified reload parity.*

---

#### Description

`kandinsky-community/kandinsky-2-2-decoder` at revision `9ae140d347fed8ce6e8bb3005dcc1f48543bb8e3` is the diffusion decoder member of the Kandinsky 2.2 latent diffusion model family (Shakhmatov et al., 2023), developed by ai-forever. It uses a 1.25 B-parameter conditional UNet architecture (`UNet2DConditionModel` with 1,224,965,124 parameters) conditioned on text-aligned CLIP image embeddings (CLIP ViT-G/14) and timesteps, operating on latents produced and decoded by a learned MoVQ autoencoder (67 M parameters). The packaged safetensors weights consist of `unet/diffusion_pytorch_model.safetensors` (5,012,319,952 bytes) and `movq/diffusion_pytorch_model.safetensors` (271,380,364 bytes).

What this repository adds is the `KandinskyPipeline` class in `src/kandinsky_generation_pipeline/pipeline.py`: manifest verification of the three Hub snapshots before any model library is imported, staging at pinned revisions, UNet construction from `diffusers` with a rank-8 LoRA injected by `peft` on attention projections (`to_q`, `to_k`, `to_v`, `to_out.0` — 176 tensors, 1,646,592 parameters), lazy loading and release of the prior pipeline with a prompt cache that outlives it, seeded generation through `KandinskyV22Pipeline`, held-out denoising-MSE evaluation at fixed timesteps, bounded LoRA fine-tuning with validation-loss epoch selection, and a safetensors adapter artifact that is digest-verified before deserialization.

#### Intended Use and Limitations

###### Primary Intended Uses

Two tasks are exposed. **Text-to-image generation:** input is text prompts (1..1,000 characters), seed, step count (1..100) and guidance scale (1..20); output is 512 × 512 RGB images decoded by MoVQ. **Adaptation to captioned images:** given `{id, image, caption}` records, the pipeline fine-tunes the LoRA on the noise-prediction objective and exports the adapter as a portable safetensors artifact.

###### Primary Intended Users

Intended users are machine-learning engineers and researchers who work with latent diffusion models and parameter-efficient fine-tuning (PEFT), in research, teaching or the DIMER model workbench.

###### Out-of-scope use cases

1. **Capability boundary:** the pipeline generates 512 × 512 images from text and fine-tunes a LoRA on the decoder UNet. It does not perform inpainting (which uses `kandinsky-inpainting-pipeline`) or ControlNet depth conditioning (which uses `kandinsky-controlnet-depth-pipeline`).
2. **Input boundary:** records are RGB images with sides within 256..4,096 px and captions of 1..1,000 characters.
3. **Decision boundary:** not for producing images presented as photographs of real events or identifiable people, or for any downstream decision that treats a generated image as factual evidence.
4. **Provenance boundary:** not for unverified execution environments; snapshots must match committed manifests.

#### Factors

###### Groups

The model generates depictions of subjects named in prompts. The upstream model was trained on diverse web image-text pairs; this repository's sample prompts focus on bird species and measure no demographic attributes. Deployments depicting people must perform domain-specific evaluation for fairness and representation.

###### Instrumentation

Training photographs came from diverse internet sources with automated and human-curated captions. Sample images are research-grade iNaturalist bird photographs. The evaluation instrument is `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0`, `diffusers==0.40.0`, `transformers==5.17.0`, `peft==0.21.0`, `accelerate==1.15.0`, `safetensors==0.8.0`, `huggingface-hub==1.32.0`, `numpy==2.5.3`, `pillow==11.3.0`. The UNet runs in float16 on CUDA with LoRA in float32 under autocast; MoVQ runs in float16. A GPU of at least 12 GB VRAM is recommended.

#### Metrics

###### Performance Measures

The pipeline reports three metric types:
1. **Held-out denoising MSE** (`evaluate`): Mean squared error of UNet noise predictions against ground-truth Gaussian noise added to MoVQ latents at timesteps 100, 300, 500, 700, 900.
2. **CLIP scores of generated images** (`score_generations`): `clip_prompt_similarity`, `label_accuracy` (nearest caption argmax), and `reference_similarity` to held-out real images.
3. **The real-photo ceiling** (`real_photo_baseline`): The same three CLIP scores evaluated directly on the held-out real photographs.

###### Decision thresholds

The generator sets no arbitrary decision threshold; the only automated rule is the CLIP `label_accuracy` nearest-caption argmax. Epoch selection chooses the checkpoint with minimal validation denoising MSE.

###### Approaches to uncertainty and variability

All generation, noise, and latent operations use seeded random generators for reproducible evaluation. Autocast and GPU attention kernel nondeterminism can produce minor variations in low-order decimals.

#### Ethical considerations and biases

###### Data

The upstream training set comprises large-scale public image datasets. This repository distributes code, manifests, and documentation only; no binary model weights or training images are stored in Git. Tutorial data uses 60 CC0 1.0 research-grade photographs from iNaturalist.

###### Human Life

This pipeline is intended for research, pedagogical, and benchmarking applications; it is not designed or validated for life-critical, medical, diagnostic, legal, or high-stakes decisions.

###### Mitigations

1. **Supply-chain integrity:** `MODEL_ID`, `PRIOR_ID`, and `SCORER_ID` are pinned to immutable 40-hex commit hashes. Manifests enforce exact byte count and SHA-256 validation before loading.
2. **Safetensors only, no remote code:** All weight files use safetensors; no pickle files are deserialized and `trust_remote_code=True` is prohibited.
3. **Prior lifecycle management:** The prior is released before training to free VRAM.
4. **Input validation:** Pre-execution checks enforce dimensions, formats, and dataset constraints.

###### Risks and harms

1. **Synthetic imagery:** The model can generate realistic synthetic images; users must label synthetic media responsibly.
2. **Bias reproduction:** Web-trained models may reflect biases present in their pre-training distributions.

###### Use cases

The model must not be used to create non-consensual sexual content, depictions of real people presented as factual, hate speech, or harassment. Use must comply with the upstream Apache-2.0 license.

## Immutable provenance

- Model: `kandinsky-community/kandinsky-2-2-decoder`
- Revision: `9ae140d347fed8ce6e8bb3005dcc1f48543bb8e3`
- Manifest: `weights/kandinsky-2-2-decoder/dimer-base-manifest.json`, format `dimer_hf_snapshot` v1, 7 files, `totalBytes` 5283700183
- UNet `unet/diffusion_pytorch_model.safetensors` (5,012,319,952 bytes) SHA-256: `9fae7efc90066b5394be5f483c65cbeeaef178a9c39c898c697845fca9b19dfb`; 1,224,965,124 parameters, float16/float32
- MoVQ `movq/diffusion_pytorch_model.safetensors` (271,380,364 bytes) SHA-256: `43a586071efc566fbceccaa3f888362629b311da70ff8ec19bebe5c48b2ddb6e`; 67 M parameters
- Prior: `kandinsky-community/kandinsky-2-2-prior` at `9fc51ad5732afc5d031724219d22e6c42179c5a8`; manifest `weights/kandinsky-2-2-prior/dimer-base-manifest.json`, 14 files, `totalBytes` 10574964619
- Scorer (evaluation only): `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` at `1a25a446712ba5ee05982a381eed697ef9b435cf`; manifest `weights/clip-vit-b-32-laion2b/dimer-base-manifest.json`, 9 files, `totalBytes` 608782299
- Upstream references: https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder · https://huggingface.co/kandinsky-community/kandinsky-2-2-prior · https://github.com/ai-forever/kandinsky-2

## Input/output contract

- `KandinskyPipeline.from_pretrained(device=None, weights_dir=None, prior_dir=None, allow_download=False, use_lora=False)`: stages and verifies snapshots, loads decoder UNet + MoVQ.
- `encode_prompts(prompts) -> dict`: loads prior, encodes text to CLIP image embeddings, caches embeddings. `release_prior() -> bool`, `export_prompt_cache() -> dict`, `import_prompt_cache(cache) -> int`.
- `generate(prompts, *, seed=0, steps=20, guidance_scale=4.0) -> dict`: returns generated PIL images and metadata.
- `evaluate(records, *, seed=0, batch_size=1) -> dict`: returns held-out denoising MSE.
- `adapt(train, val=None, *, epochs=4, lr=1e-4, batch_size=1, seed=0) -> dict`: bounded AdamW LoRA fine-tuning.
- `save_artifact(output_dir, metadata=None) -> Path`: writes `adapter.safetensors` (176 LoRA tensors) + `manifest.json`.

## DIMER deployment notes

| Field | Status |
|---|---|
| **DIMER status** | Planned / Tier D GEN row |
| Licence | Apache-2.0 for decoder and prior; MIT for scorer; code Apache-2.0 |
| Weights | 15.9 GB total (>9 GB publication gate waived per PixArt-Σ / Toto precedent) |
| Remote code | Not required — standard diffusers and transformers classes |
| Executable serialization | None — safetensors only |
| Runtime | PyTorch 2.14+, diffusers, transformers, peft; fp16 on CUDA |

## Runtime

- Local feasibility gate executed 2026-09-24 on NVIDIA GeForce RTX 5070 Ti Laptop GPU (12.2 GB usable VRAM in WSL):
  - UNet inference: 20 steps @ 512² in 14.70 s wall time.
  - Peak VRAM allocated: 3,224.4 MB (reserved: 3,904.0 MB).
  - CPU offload needed: False.
- Unit test suite: 34 tests passing in 10.36 s offline.
- Notebook specification: DIMER Notebook Specification 2.0 (§4 standalone carrier).

## References

- Shakhmatov, A., et al. (2023). Kandinsky 2.2. ai-forever. https://github.com/ai-forever/kandinsky-2
- Rombach, R., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. CVPR.
- Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.
- Pinned repositories: https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder · https://huggingface.co/kandinsky-community/kandinsky-2-2-prior · https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K
