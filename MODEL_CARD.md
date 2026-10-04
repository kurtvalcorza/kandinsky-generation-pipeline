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
> **Three pinned snapshots make one generator, the prior pipeline can be released before training, and generation has no ground truth.** The decoder checkpoint ships the 5.01 GB UNet and 0.27 GB MoVQ; the prior repository provides the CLIP text and image encoders and prior diffusion transformer, and a CLIP ViT-B/32 is pinned for evaluation only. `src/kandinsky_generation_pipeline/pipeline.py` stages and digest-verifies all three, encodes prompts once and releases the prior pipeline before training, and scores the model with a held-out denoising loss and CLIP scores beside a leave-one-out real-photo reference line (not a ceiling).

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run, self-contained Google Colab notebook. It carries the repository's code in its own cells and runs end to end without cloning the repository. It installs nothing into the notebook kernel: every stage runs in an isolated environment built from a committed hash lock, so `Run all` needs no runtime restart:

- **End-to-End Text-to-Image Fine-Tuning Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/kandinsky-generation-pipeline/blob/main/tutorials/kandinsky_generation_colab.ipynb) [`kandinsky_generation_colab.ipynb`](https://github.com/kurtvalcorza/kandinsky-generation-pipeline/blob/main/tutorials/kandinsky_generation_colab.ipynb)  
  *End-to-end LoRA fine-tuning of Kandinsky 2.2 with pinned snapshots: 60 digest-pinned CC0 iNaturalist bird photographs with a seeded stratified split, structural validation with refusal probes, prompt encoding with the prior pipeline released before training, held-out denoising loss and CLIP-scored generations beside a leave-one-out real-photo reference line for the frozen model, bounded LoRA fine-tuning of the UNet attention projections, a paired comparison on identical held-out inputs, a new prompt rendered, and safetensors adapter export with verified reload parity.*

---

#### Description

`kandinsky-community/kandinsky-2-2-decoder` at revision `9ae140d347fed8ce6e8bb3005dcc1f48543bb8e3` is the diffusion decoder of Kandinsky 2.2, a latent diffusion text-to-image model from ai-forever (Shakhmatov et al., 2023). Generation runs in two stages. First, the separately pinned prior `kandinsky-community/kandinsky-2-2-prior` maps a text prompt to a CLIP ViT-G/14 image embedding. Second, the decoder's conditional UNet (`UNet2DConditionModel`, 1,253,057,288 parameters) removes noise from a 64 × 64 × 4 latent over a fixed number of scheduler steps, conditioned on that embedding. A MoVQ autoencoder (67,832,495 parameters) decodes the final latent to a 512 × 512 RGB image.

The upstream weights do not change at inference time. Adaptation in this repository is gradient training of a rank-8 LoRA adapter (`peft`) on the UNet attention projections `to_q`, `to_k`, `to_v` and `to_out.0`: 176 tensors, 1,646,592 trainable parameters. The prior, the MoVQ and the base UNet weights stay frozen.

This repository adds the `KandinskyPipeline` class in `src/kandinsky_generation_pipeline/pipeline.py`, the captioned-image contract in `samples.py`, and the CLIP-based scoring in `metrics.py`. When a reader runs this code, it does the following:

- verifies three Hub snapshots against committed manifests before any model library is imported;
- encodes prompts through the prior, then releases the prior;
- generates seeded images;
- evaluates held-out denoising MSE;
- fine-tunes the LoRA adapter with validation-loss epoch selection;
- writes a safetensors adapter artifact whose digest is checked before it is loaded again.

#### Intended Use and Limitations

The pipeline is a teaching and research reference for adapting a latent diffusion generator to a small captioned image set.

###### Primary Intended Uses

The pipeline exposes two tasks:

- **Text-to-image generation.** The input is one or more text prompts (1..1,000 characters each), a seed, a step count (1..100, default `20`) and a guidance scale (1..20, default `4.0`). The output is one 512 × 512 RGB image per prompt and seed, plus generation metadata.
- **Adaptation to captioned images.** The input is a list of `{id, image, caption}` records (4..2,000 records). The output is a LoRA adapter trained on the noise-prediction objective, written as `adapter.safetensors` with a `manifest.json`.

The application domains envisioned during development are these:

- teaching how parameter-efficient fine-tuning changes a diffusion model;
- research on the style or subject adaptation of a text-to-image generator to a narrow, captioned photo collection, such as a set of species photographs;
- producing illustrative synthetic images for such research.

The design role is a reference implementation that a reader's own training or generation code can embed or copy. It is not a production image service.

###### Primary Intended Users

The intended users are machine-learning engineers, researchers and students who work with diffusion models and parameter-efficient fine-tuning. The envisioned settings are research, teaching and self-hosted experimentation on a GPU the user controls.

The pipeline assumes the following user competencies:

- how diffusion sampling and classifier-free guidance affect an image;
- that a lower denoising loss does not imply better-looking images;
- that CLIP similarity is an automated proxy and not a human judgement;
- how to check the licence and consent status of the images they train on.

The pipeline enforces only structural limits on records and prompts. It cannot judge whether an image or caption is appropriate, so it is not robust to careless or adversarial inputs.

###### Out-of-scope use cases

1. **Capability boundary:** the pipeline generates 512 × 512 images from text and fine-tunes a LoRA on the decoder UNet. It does not perform inpainting or depth-conditioned generation, and it does not generate at other resolutions. Masked editing and depth-conditioned generation belong to separate Kandinsky 2.2 inpainting and ControlNet-depth pipelines, which are still under development.
2. **Input boundary:** training records must be images whose sides are within 256..4,096 px, with captions of 1..1,000 characters. A dataset must have 4..2,000 records, and record ids must be unique strings of at most 64 characters. Each image is resized so its shorter side is 512 px and centre-cropped to 512 × 512, so content outside the central square is not learned. Prompts are refused outside 1..1,000 characters, steps outside 1..100 and guidance outside 1..20.
3. **Data-size boundary:** the adaptation is designed for tens to hundreds of images and at most 50 epochs. It is not a full fine-tuning recipe, and it is not tested on thousands of images.
4. **Decision boundary:** not for producing images presented as photographs of real events or identifiable people. Not for any decision that treats a generated image as factual evidence, such as news, legal, insurance or scientific evidence.
5. **Provenance boundary:** not for use with weights that do not match the committed manifests. The loader refuses a mismatch rather than loading it.

---

#### Factors

The pipeline's behaviour varies with the subject named in the prompt, the photographs used for adaptation, and the hardware it runs on.

###### Groups

The pipeline is not human-centric by design. The bundled sample contains photographs of six North American bird species, and the evaluation measures no demographic attribute. The model can still depict people when a prompt asks for them. The upstream training corpus is web image–text data that the upstream authors did not audit or document by demographic group. It is unknown how people of different ages, genders, skin tones or cultures are represented in it.

This repository therefore makes no group-level fairness claim. An operator who generates or adapts images depicting people takes on that audit. The operator should compare generated depictions across the groups relevant to their use, for example by fixed prompts that vary only the described group, before relying on the output.

###### Instrumentation

The upstream model was trained on image–text pairs collected from the web, captured by unknown cameras and captioned by their publishers or by automated tools. The upstream authors do not document the capture instruments. The tutorial sample consists of 60 research-grade iNaturalist photographs. They were taken by volunteer observers, typically with consumer cameras and phones, and vary in resolution, focus and lighting.

The pipeline sees only decoded RGB pixels after a resize and centre crop. It cannot detect instrument defects such as blur, compression artefacts, watermarks or colour casts. An adapter trained on photographs with such defects learns to reproduce them. Captions are also an instrument: the sample captions come from one fixed template per species, so they carry no description of pose, background or lighting.

###### Environment

**Operating environment.** Python 3.12 with the pinned packages `torch==2.14.0`, `diffusers==0.40.0`, `transformers==5.17.0`, `peft==0.21.0`, `accelerate==1.15.0`, `safetensors==0.8.0`, `huggingface-hub==1.32.0`, `numpy==2.5.3` and `pillow==11.3.0`. The tutorial notebook does not install these into its kernel: it builds a separate CPython 3.12.12 environment with a pinned `uv` from `tutorials/requirements-colab.lock.txt`, which locks those pins and all their dependencies to exact versions and SHA-256 digests for Linux x86_64 (the Linux `torch` 2.14.0 wheel is the CUDA 13.0 build), and runs each stage in its own process there. The UNet, prior and MoVQ run in float16 on CUDA. The LoRA parameters are kept in float32, and training uses float16 autocast. The tutorial requires a Linux x86_64 runtime with a CUDA GPU of at least 15 GB of memory, such as a 16 GB T4, and about 30 GB of free disk: 16.5 GB of pinned weights and about 12 GB for the isolated environment. On CPU the code runs in float32 but is impractically slow for the tutorial.

**Data environment.** Adaptation assumes that the photographs the adapter is trained on resemble the images the user later wants to generate: the same kind of subject, framing and photographic style. The captions used at generation time should follow the same pattern as the training captions. Prompts that fall outside the adapted subject revert towards the base model's behaviour. Held-out denoising loss is meaningful only when the validation photographs come from the same distribution as the training photographs.

---

#### Metrics

The metrics are chosen because generation has no ground truth. Each metric measures one property of the model, and none of them measures image quality as a person would judge it.

###### Performance Measures

The code reports these measures. Names are given as the code reports them.

1. **`denoising_mse`** from `evaluate`: the mean squared error between the UNet's noise prediction and the true Gaussian noise added to MoVQ latents of held-out photographs, at timesteps `100`, `300`, `500`, `700` and `900`, with a per-timestep breakdown. It is the training objective measured on photographs the adapter never trained on, and it is the only measure that compares the frozen and the adapted model on identical inputs.
2. **`clip_prompt_similarity`** from `score_generations`: the mean cosine similarity × 100 between each generated image and its prompt, embedded by the pinned CLIP ViT-B/32 scorer. It captures prompt alignment.
3. **`label_accuracy`** from `score_generations`: the fraction of generated images whose nearest caption, among the dataset's distinct captions, is the caption they were generated from. It captures whether a generated image is recognisable as its intended subject.
4. **`reference_similarity`** from `score_generations`: the mean cosine similarity × 100 between each generated image and the mean CLIP embedding of held-out real photographs with the same caption. It captures closeness to the target photographs.
5. **`real_photo_reference`** (alias `real_photo_baseline`): the same three CLIP measures computed on the held-out real photographs, each photo's reference similarity measured against the other held-out photos of its caption, never itself (leave-one-out; a caption with one held-out photo gets no value). It is a reference line, not a ceiling: a generator conditioned on the prompt can score above it on prompt similarity and reference similarity. The public sample photographs may also overlap the generator's and the scorer's web-scale pretraining data, so these values on the sample may be optimistic.

The measures are complementary. The denoising loss is sensitive to the adaptation but does not show what images look like. The CLIP measures describe generated images but depend on one automated scorer and on the prompt wording. Reading only the loss would miss a model that fits noise better but generates worse images. Reading only CLIP similarity would reward images that match the text but not the photographs. No FID or human preference score is computed. The 12 held-out images are far too few for FID.

###### Decision thresholds

The pipeline applies two implicit decision rules:

- `label_accuracy` uses an `argmax` over the CLIP cosine similarities to the distinct captions, with no minimum similarity.
- Adaptation keeps the epoch with the lowest validation `denoising_mse`, including epoch 0, the frozen model. The kept adapter can therefore never have a higher validation loss than the frozen model.

No acceptance threshold on any measure was set during development, and no quality threshold is shipped. A similarity score or loss value from this pipeline cannot, on its own, decide whether an image is fit for use. The operator who deploys generated images owns any acceptance rule. A false accept, where an unsuitable or misleading image is published, usually costs more than a false reject, where a usable image is discarded. The operator should therefore combine automated scores with human review, and set any threshold on their own validation images.

###### Approaches to uncertainty and variability

Every reported number comes from a single run on one seeded split of the tutorial sample: 36 training, 12 validation and 12 test photographs, 6, 2 and 2 per species with seed `42`. No repeated runs, cross-validation or bootstrap are performed, and no standard deviation or confidence interval is reported. Differences between the frozen and the adapted model are one observation on 12 photographs and 12 generated images, not an estimate of a population effect.

Seeds control the prompt encoding, the data split, the evaluation noise and latents (derived from the evaluation `seed` and timestep), the training noise and ordering (`seed`), and each generated image (`seed + index`). The remaining sources of run-to-run variability are as follows:

- **Non-deterministic GPU kernels and float16 autocast** can change low-order digits between runs on the same hardware, and more between different GPUs.
- **Prompt encoding** seeds the prior's sampler with `prompt_seed(prompt)`, a 31-bit integer taken from the prompt's SHA-256. The same prompt therefore gets the same prior seed in every process. Its embedding is still subject to the GPU nondeterminism above.

CLIP similarities are cosine similarities, not probabilities, and `label_accuracy` is not calibrated. A caller who needs a calibrated measure must label their own images and calibrate against them.

---

#### Ethical considerations and biases

No external ethics board or group review has assessed this pipeline. The considerations below are the developers' own.

###### Data

The upstream Kandinsky 2.2 model was trained on large web-scale image–text datasets. The upstream authors describe these only at a high level and do not list their sources or filtering in the model repository. It is therefore not ruled out that the training data includes personal images, faces, copyrighted works or other sensitive material. Whether it does is unknown.

This repository distributes code, snapshot manifests, configuration files and documentation. It does not distribute model weights in Git: they are fetched from the Hugging Face Hub at the pinned revisions. It also does not distribute training images. The tutorial downloads 60 CC0 1.0 photographs from the iNaturalist open-data bucket at run time, and the adapter it exports is trained only on those.

The operator is responsible for the images and captions they supply for adaptation. The pipeline does not check them for faces, personal data, copyrighted content or confidential material. An adapter can memorise and reproduce its training images, so an adapter trained on restricted images must be treated as restricted too.

###### Human Life

The pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, housing or any other domain central to human life. It produces synthetic images, and no generated image should be treated as a record of a real person, place or event. Nobody has validated it for any such domain. Its only checks are structural tests and a tutorial run on bird photographs.

Foreseeable misuse in a sensitive domain includes generating images that could be mistaken for medical, forensic or news imagery. Such use would require, at minimum, human review of every image, disclosure that the image is synthetic, and validation by the responsible domain authority. This repository provides none of these.

###### Mitigations

1. **Supply-chain integrity:** `MODEL_REVISION`, `PRIOR_REVISION` and `SCORER_REVISION` are immutable 40-character commit hashes. Each snapshot is checked against its committed `dimer-base-manifest.json`, and every file's byte count and SHA-256 must match before loading. Staging refuses a manifest that names a different model or revision, and downloads only at the pinned revision.
2. **No executable serialization:** every weight file is safetensors, and the snapshot check rejects file types outside the manifest's code-free set. No pickle is deserialized, and no Hub-hosted code runs: the model classes come from `diffusers`, `transformers` and `peft`.
3. **Adapter integrity:** `load_adapter` refuses an artifact whose `format` is not `org.valcorza.kandinsky-generation.adapter.v1`, whose recorded base model is not the pinned decoder revision, or whose `adapter.safetensors` SHA-256 differs from its manifest.
4. **Input integrity:** `validate_dataset` rejects datasets outside 4..2,000 records, duplicate ids, missing fields, image sides outside 256..4,096 px and captions outside 1..1,000 characters, before any model runs. `generate` rejects out-of-range steps and guidance.
5. **Bounded adaptation:** `adapt` refuses more than 50 epochs or a learning rate above `1e-2`, and keeps the epoch with the lowest validation loss, so an adaptation that makes the model worse on held-out data is not exported.
6. **Reproducibility:** the split, the training and evaluation noise, and each generated image are seeded. Runtime packages are pinned exactly in `pyproject.toml`; the notebook installs them, with every transitive dependency, from a hash lock into an isolated environment and never into the hosted runtime's own interpreter. The exported manifest records the base model identity and adapter configuration. Prompt encoding is seeded from a SHA-256 of the prompt, so it does not depend on Python's per-process hash salt.

The pipeline has no content filter or safety checker on prompts or generated images, and it adds no watermark or provenance metadata to generated images.

###### Risks and harms

1. **Misleading synthetic imagery.** The model produces photorealistic images of things that did not happen. Third parties who see an image without disclosure bear the harm, and the operator bears the reputational harm. The risk is realised whenever generated images are shared without a synthetic label, and it is likely under normal use because this pipeline adds no watermark. The magnitude ranges from minor confusion to serious harm when an image is used as evidence.
2. **Harmful or non-consensual content.** No content filter runs, so a prompt can produce violent, sexual or defamatory depictions, including of real people. The people depicted bear the harm. Its likelihood depends on who can submit prompts, and its magnitude can be severe.
3. **Bias amplification.** Web-trained generators reproduce stereotypes in how they depict people, occupations and cultures, and an adapter trained on a skewed set of images narrows the output further. The groups depicted and the viewers bear the harm. It is likely whenever people are generated without the audit described under *Groups*.
4. **Training-data leakage.** A LoRA trained on a few images can reproduce them closely. If those images are private or copyrighted, the adapter and its outputs can leak them. The data subjects and rights holders bear the harm, and it is likely with small datasets and many epochs.
5. **Automation bias.** Users may read a rising CLIP score or a falling denoising loss as proof of better images. The operator then accepts worse outputs, which the downstream audience bears. It is likely when scores are reported without human review.
6. **Out-of-distribution degradation.** Prompts far from the adapted subject produce images of unknown quality with no warning. The operator bears the harm.

###### Use cases

The following uses are unacceptable even where the pipeline would work:

1. creating sexual or intimate imagery of real people, or any sexual imagery of minors;
2. creating images of real people or events presented as authentic, including disinformation, fabricated evidence and impersonation;
3. generating harassment, hate imagery or material intended to intimidate or demean a person or group;
4. surveillance, biometric identification or demographic profiling, and training adapters on photographs of people collected without their consent;
5. producing images used to discriminate in employment, housing, credit, insurance, education or healthcare access;
6. deceptive, manipulative or fraudulent applications, such as fake product photographs or fake identity documents;
7. any use that violates the upstream Apache-2.0 licence, the rights attached to the training images, or applicable law.

## Immutable provenance

- Model: `kandinsky-community/kandinsky-2-2-decoder`
- Revision: `9ae140d347fed8ce6e8bb3005dcc1f48543bb8e3`
- Manifest: `weights/kandinsky-2-2-decoder/dimer-base-manifest.json`, format `dimer_hf_snapshot` v1, 7 files, `totalBytes` 5283700183
- UNet `unet/diffusion_pytorch_model.safetensors` (5,012,309,584 bytes) SHA-256: `3cc2f07442b9de0f18fb3f22247790872c094c4acc9dbe68552f96fd2e5c1ea1`; 1,253,057,288 parameters, float16/float32
- MoVQ `movq/diffusion_pytorch_model.safetensors` (271,380,364 bytes) SHA-256: `43a5860fea195a7116f2471396c5cc9535fade9b63c4857d8a192ffd924b7002`; 67 M parameters
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

## Deployment notes

| Field | Status |
|---|---|
| Licence | Apache-2.0 for decoder and prior; MIT for scorer; code Apache-2.0 |
| Weights | About 15.9 GB served (5.28 GB decoder and 10.57 GB prior), plus the 0.6 GB evaluation scorer |
| Remote code | Not required: standard `diffusers` and `transformers` classes |
| Executable serialization | None: safetensors only |
| Runtime | PyTorch 2.14, `diffusers`, `transformers`, `peft`; float16 on CUDA |

## Verification records

`docs/release-verification.md` holds the procedure and every record. The current notebook, which runs every stage in an isolated hash-locked environment, was run three times at commit `256fcb2`:

- **Date:** 2026-09-30
- **Subject:** `tutorials/kandinsky_generation_colab.ipynb` at commit `256fcb2`, blob `26a6d01839ff`
- **Runtime:** Google Colab, Tesla T4 (15,360 MiB); the notebook kernel ran Python 3.13.15, and the isolated environment ran Python 3.12.12 with `torch 2.14.0+cu130`, `diffusers 0.40.0`, `transformers 5.17.0` and `peft 0.21.0`
- **Procedure:** `Run all` from a fresh runtime with the form fields at their defaults
- **Observed result:** 11 of 11 code cells ran in one pass without error or restart. Held-out test `denoising_mse` was 0.077038 for the frozen model and 0.07613 after adaptation; `label_accuracy` was 0.8333 for both, against 0.9167 for the real photographs. A reload in a fresh process reproduced the adapted model's results exactly.
- **Caveats:** one run on one seeded split. This is sample-sanity evidence, not a benchmark.

The same commit also passed a Kaggle T4 run in strict single-pass mode (a restart request fails the run), with the same results, and the BYOD journey: 12 representative photographs were carried through every stage, and a `captions.csv` without its `caption` column and a 200 × 200 image were each refused with the validator's message.

`docs/release-verification.md` also keeps the records of the **previous** notebook revision, which installed its pins into the kernel and needed a manual restart on hosted runtimes. The clean-runtime run of that previous tutorial notebook:

- **Date:** 2026-09-29
- **Subject:** `tutorials/kandinsky_generation_colab.ipynb` at commit `b674640`, blob `34711f6de24f` (full identifiers in `docs/release-verification.md`)
- **Runtime:** Kaggle batch kernel on a Tesla T4 (15,360 MiB), Python 3.12.13, `torch 2.14.0+cu130`, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0`
- **Procedure:** the notebook was fetched at that commit and run with `Run all` in a fresh interpreter, with an empty Hugging Face cache and no repository checkout, once with the form fields at their defaults. The install cell's restart guard fired once because the kernel had preloaded older `numpy` and `protobuf`, and the kernel was restarted and run again from the top.
- **Observed result:** 12 of 12 code cells ran without error in 906.8 s. Held-out test `denoising_mse` was 0.077038 for the frozen model and 0.076142 after adaptation. `label_accuracy` was 0.8333 for both, against 0.9167 for the real photographs. `reference_similarity` rose from 68.910 to 69.087, against 88.660 for the real photographs. The reloaded adapter reproduced the in-memory results exactly: `denoising_mse_diff` 0.0 and `mean_abs_pixel_diff` 0.0.
- **Caveats:** one run on one seeded split with 12 held-out photographs and 12 generated images. This is sample-sanity evidence, not a benchmark.

The BYOD branch was run at the same commit:

- **Date:** 2026-09-29
- **Subject:** the same notebook and commit, with `USE_BYOD = True` and `BYOD_PATH` set in the executed copy only
- **Runtime:** as above
- **Procedure:** a zip of 12 CC0 research-grade iNaturalist photographs (6 Northern Cardinal, 6 Blue Jay) with a `captions.csv` was built inside the kernel, each photograph checked against a pinned SHA-256. After `Run all`, the committed Section 4 source was re-run against two incompatible zips.
- **Observed result:** 13 of 13 code cells ran without error in 688.7 s. The 12 records were split 8 / 2 / 2 by caption and passed through fine-tuning, evaluation, export and an exact reload. A `captions.csv` without its `caption` column and a 200 × 200 image were each refused with a message naming the failed rule, before any model ran on them.
- **Caveats:** with one test photograph per caption, these numbers show that the BYOD path runs, not how well the model adapts to such data.

An earlier pre-flight observation of the generator, not the notebook:

- **Date:** 2026-09-24
- **Subject:** the `KandinskyPipeline` generation path in `src/kandinsky_generation_pipeline/pipeline.py`, before the weight-facts corrections in commit `ec6d560`
- **Runtime:** NVIDIA GeForce RTX 5070 Ti Laptop GPU (12.2 GB usable), CPython 3.12.3, the pinned `torch`, `diffusers` and `transformers` versions
- **Procedure:** staged and verified the decoder and prior; generated one 512 × 512 image with 20 steps and guidance `4.0`
- **Observed result:** 14.70 s wall time for the 20 steps; peak allocated GPU memory 3,224.4 MB (3,904.0 MB reserved); no CPU offload needed
- **Caveats:** one generation on one workstation with weights pre-staged; not a run of the tutorial notebook, not training, and not a clean runtime

The offline unit tests and `tools/validate_release_assets.py` are static and unit checks, not executions of the notebook. The tutorial follows DIMER Notebook Specification 2.2 as a standalone notebook.

## References

- Shakhmatov, A., et al. (2023). Kandinsky 2.2. ai-forever. https://github.com/ai-forever/kandinsky-2
- Rombach, R., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. CVPR.
- Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.
- Pinned repositories: https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder · https://huggingface.co/kandinsky-community/kandinsky-2-2-prior · https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K
