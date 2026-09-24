# Kandinsky 2.2 Generation Pipeline

DIMER-oriented pipeline for **Kandinsky 2.2 Text-to-Image** (`kandinsky-community/kandinsky-2-2-decoder`, the 1.25 B-parameter UNet diffusion model for 512 × 512 and 1024 × 1024 text-to-image generation), pinned to an immutable Hugging Face revision together with the shared `kandinsky-community/kandinsky-2-2-prior` and a CLIP scorer for evaluation. The repository exposes seeded text-to-image generation, a held-out denoising-loss and CLIP-scored evaluation against a real-photo ceiling, a captioned-image contract with explicit ceilings, a bounded LoRA fine-tuning contract with a portable safetensors adapter, a `MODEL_CARD.md` at DIMER Model Card Specification 1.2, and a standalone `E2E` tutorial at DIMER Notebook Specification 2.0.

## Upstream alignment

- Model: `kandinsky-community/kandinsky-2-2-decoder`
- Revision: `9ae140d347fed8ce6e8bb3005dcc1f48543bb8e3`
- Prior: `kandinsky-community/kandinsky-2-2-prior` at `9fc51ad5732afc5d031724219d22e6c42179c5a8` (shared prior model, CLIP text encoder, CLIP ViT-G image encoder); scorer `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` at `1a25a446712ba5ee05982a381eed697ef9b435cf` (evaluation only)
- Upstream weight license: **Apache-2.0** for both decoder and prior; MIT for the scorer
- Upstream task: text-to-image generation at 512 × 512 — a two-stage latent diffusion system: prior maps text to CLIP image embeddings, and decoder UNet denoises latents conditioned on image embeddings
- Runtime: `diffusers==0.40.0` + `transformers==5.17.0` + `peft==0.21.0` + `torch==2.14.0` — every weight file is safetensors, **nothing is unpickled and no Hub-hosted code is executed**
- Repository adaptation: **E2E** (bounded LoRA fine-tuning of the UNet attention projections on captioned images, with a portable safetensors adapter)

## Two things to know before you start

**The prior pipeline is loaded once, used once and released.** The prior pipeline takes ~7 GB in float16 — releasing it before training leaves the GPU with maximum VRAM headroom for the UNet training graph. `encode_prompts()` encodes text prompts into CLIP image embeddings, and `release_prior()` drops the prior; `generate()`, `evaluate()` and `adapt()` read the prompt cache, and `export_prompt_cache()` / `import_prompt_cache()` hand it to a second pipeline without reloading the prior.

**Generation has no ground truth, and the numbers say what they are.** `evaluate()` reports the held-out *denoising loss* — the training objective on photographs the model never trained on, at fixed timesteps with seeded noise, so the frozen and adapted models see identical inputs. `metrics.score_generations()` reports CLIP prompt similarity, label accuracy, and reference similarity to real photographs; `real_photo_baseline()` reports the same three numbers on real photographs — the ceiling. None of these is a human judgement of image quality.

## Quick start

```python
from kandinsky_generation_pipeline import KandinskyPipeline, fetch_sample_dataset, sample_prompts
from kandinsky_generation_pipeline.metrics import ClipScorer, score_generations, real_photo_baseline

pipe = KandinskyPipeline.from_pretrained(use_lora=True)     # verifies snapshots, loads decoder UNet + LoRA, MoVQ
splits = fetch_sample_dataset()                              # 36 / 12 / 12 pinned CC0 iNaturalist bird photographs, six captions
pipe.encode_prompts(sample_prompts(splits["train"] + splits["validation"] + splits["test"]))
pipe.release_prior()                                         # prior released; image embeddings cached
print(pipe.evaluate(splits["test"])["denoising_mse"])       # frozen model, held-out denoising loss
scorer = ClipScorer()
images = pipe.generate(sample_prompts(splits["test"]), seed=1000)["images"]
print(score_generations(scorer, images, references=splits["test"]), real_photo_baseline(scorer, splits["test"]))
pipe.adapt(splits["train"], splits["validation"])            # bounded LoRA fine-tuning, epoch selected by validation loss
print(pipe.evaluate(splits["test"])["denoising_mse"])       # adapted model, identical inputs
pipe.save_artifact("outputs/adapter")
```

## Weights layout

```
weights/kandinsky-2-2-decoder/    README.md  model_index.json  scheduler/  movq/  unet/  dimer-base-manifest.json
                                  unet/diffusion_pytorch_model.safetensors  (git-ignored, 5.01 GB)
                                  movq/diffusion_pytorch_model.safetensors  (git-ignored, 271 MB)
weights/kandinsky-2-2-prior/      README.md  model_index.json  prior/  image_encoder/  text_encoder/  scheduler/
                                  dimer-base-manifest.json  (git-ignored, 10.57 GB)
weights/clip-vit-b-32-laion2b/    config, tokenizer and preprocessor files  dimer-base-manifest.json  model.safetensors  (git-ignored, 605 MB)
weights/inat-birds/               the 60 pinned photographs, cached on first fetch (git-ignored)
```

`from_pretrained()` calls `stage_missing_files()` and `stage_missing_prior_files()` (fetch only absent manifest entries at pinned revisions with `allow_download=True`), then `verify_snapshot()` and `verify_prior_snapshot()` (byte size + SHA-256 of every entry). `docs/WEIGHTS.md` records the provenance of all snapshots and the large-weights gate waiver.

## Sample data

`fetch_sample_dataset()` fetches 60 research-grade iNaturalist photographs of six North American birds (10 per species, CC0 1.0) from the public open-data bucket, each pinned by byte size and SHA-256 in `SAMPLE_RECORDS`. Captions come from one template per species. `build_sample_dataset` draws a seeded 6 / 2 / 2 stratified split per species (36 / 12 / 12) and `check_split_disjoint` asserts no image appears twice. `write_dataset_csv` / `load_byod_dataset` support BYOD workflows.

## Adapter artifacts

`save_artifact(dir)` writes `adapter.safetensors` (the 176 LoRA tensors — rank 8 on `to_q`, `to_k`, `to_v`, `to_out.0` of attention blocks, 1,646,592 parameters) and `manifest.json`. `KandinskyPipeline.from_artifact(dir)` re-verifies snapshots, checks the manifest, scope and digest before deserialising, and rebuilds the pipeline with the trained adapter.

## Tests

```
pip install -e . --no-deps
pytest
```

Tests run offline: temporary manifests, synthetic images, stub UNet, never requiring network weights or heavyweight libraries.

## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/kandinsky-generation-pipeline/blob/main/tutorials/kandinsky_generation_colab.ipynb)

`tutorials/kandinsky_generation_colab.ipynb` is declared `E2E` and is **standalone** (DIMER Notebook Specification 2.0 §4): it is generated by `tools/build_notebook.py` from `tools/notebook_template.py` and embeds the 3 package modules (`pipeline.py`, `samples.py`, `metrics.py`) verbatim in dependency order, the three pinned identities and manifests, and runtime pins.

## Release status

**Candidate** — the `E2E` tutorial notebook `tutorials/kandinsky_generation_colab.ipynb` is generated and verified with all static, parity, unit and local GPU pre-flight tests passing. Promotion to Release-grade is documented in `docs/release-verification.md` and `STATUS.md`.

## Licensing

- Upstream weights: Apache-2.0 (`kandinsky-community/kandinsky-2-2-decoder` and `kandinsky-community/kandinsky-2-2-prior`). Scorer is MIT.
- Tutorial data: iNaturalist research-grade photographs, each CC0 1.0 (observers credited in `samples.py`).
- This repository's code and documentation: Apache-2.0 (`LICENSE`).

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
