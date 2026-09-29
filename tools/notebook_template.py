"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.2 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, samples.py, metrics.py), and the model pin/stage/verify cells are produced
by the generator from repository sources so they cannot drift from the package.

This template configures an E2E text-to-image fine-tuning workflow: the pinned Kandinsky 2.2 decoder,
its shared diffusion prior and the CLIP scorer are staged and digest-verified, 60 pinned CC0
iNaturalist bird photographs are fetched, validated and split, every prompt is encoded once into
image embeddings with the prior and the prior is released, the frozen model is scored (held-out denoising
loss, CLIP-scored generations) against the real-photo ceiling, a bounded LoRA fine-tuning runs in the kernel,
the held-out scores are read again in a paired comparison, a new prompt is rendered, and the adapter is
exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "kandinsky-generation-pipeline"

BADGES = [
    (
        "GitHub",
        "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
        f"https://github.com/kurtvalcorza/{REPO}",
    ),
    (
        "Open In Colab",
        "https://colab.research.google.com/assets/colab-badge.svg",
        f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/kandinsky_generation_colab.ipynb",
    ),
    (
        "Hugging Face",
        "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-kandinsky--community%2Fkandinsky--2--2--decoder-ffcc4d?style=flat",
        "https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder",
    ),
    (
        "Upstream",
        "https://img.shields.io/badge/Upstream-ai--forever%2FKandinsky--2-181717?style=flat&logo=github&logoColor=white",
        "https://github.com/ai-forever/Kandinsky-2",
    ),
    ("License", "https://img.shields.io/badge/License-Apache--2.0-blue.svg", "https://www.apache.org/licenses/LICENSE-2.0"),
]

TEMPLATE = {
    "package": "kandinsky_generation_pipeline",
    "repo_name": REPO,
    "stem": "kandinsky_generation",
    "notebook_name": "kandinsky_generation_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh **GPU** runtime (a 16 GB T4 is enough; see the Prerequisites) installs the pinned "
        "dependencies (torch, diffusers, transformers, peft, accelerate, sentencepiece, safetensors, huggingface-hub, numpy, "
        "pillow), stages and digest-verifies three pinned snapshots from the Hub — the 5.28 GB Kandinsky 2.2 decoder (UNet + MoVQ), "
        "the 10.57 GB Kandinsky 2.2 diffusion prior, and a 0.6 GB CLIP scorer — loads the UNet in float16 with an untrained LoRA "
        "adapter attached, fetches 60 CC0 iNaturalist bird photographs as digest-verified JPEGs (6 MB, no credential), validates "
        "them and splits them 36 / 12 / 12 by seed, encodes every prompt with the prior into image embeddings and releases the prior, "
        "scores the frozen model — the held-out denoising loss on the validation and test photographs, and twelve generated images "
        "scored by CLIP against their prompts, the held-out photographs and the real-photo ceiling — runs a bounded LoRA fine-tuning "
        "(4 epochs over 36 images), scores the adapted model on identical inputs, renders a new prompt, exports the adapter as "
        "safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify generation parity. The default path "
        "needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit "
        "(NOTEBOOK_SPEC 2.2 §5). On a T4 the whole path takes about fifteen minutes of model time after the 16.5 GB of downloads."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "captioned photographs as a zip holding `captions.csv` (columns `id`, `file`, `caption`) beside the image files (JPEG or "
        "PNG, shorter side 256..4096 px; at least four images, and at least one caption with three or more images so a held-out "
        "record exists). Your records are split by caption into training, validation and test sets and flow through the same "
        "contract — validation, prompt encoding, frozen baseline, adaptation, held-out evaluation, generation, artifact export "
        "and reload parity. The expected schema, the ceilings and the privacy guidance are stated in the Prerequisites and in "
        "Section 4, and uploaded files stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "pipeline_class": "KandinskyPipeline",
    "model_load": "KandinskyPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, prior_dir=PRIOR_WEIGHTS_DIR, device=('cuda' if torch.cuda.is_available() else 'cpu'), use_lora=True)",
    "weights_key": "kandinsky-2-2-decoder",
    "modules": ["pipeline.py", "samples.py", "metrics.py"],
    "entry_module": "pipeline.py",
    "rewrites": [
        [
            r"^_WEIGHTS_ROOT = Path\(__file__\)[^\n]*$",
            '_WEIGHTS_ROOT = Path.cwd() / "weights"  # standalone rewrite (build_notebook.py): working-directory-relative',
        ]
    ],
    "extra_weights": [
        {
            "key": "kandinsky-2-2-prior",
            "var": "PRIOR_MANIFEST",
            "dir": "PRIOR_WEIGHTS_DIR",
            "identity": ["PRIOR_ID", "PRIOR_REVISION"],
            "stage": "stage_missing_prior_files",
            "verify": "verify_prior_snapshot",
        },
        {
            "key": "clip-vit-b-32-laion2b",
            "var": "SCORER_MANIFEST",
            "dir": "SCORER_WEIGHTS_DIR",
            "identity": ["SCORER_ID", "SCORER_REVISION"],
            "stage": "stage_missing_scorer_files",
            "verify": "verify_scorer_snapshot",
        },
    ],
    "runtime_imports": ["torch", "diffusers", "transformers", "peft"],
    "title": "Kandinsky 2.2 — DIMER E2E text-to-image fine-tuning tutorial (standalone)",
    "badges": BADGES,
    "capability": "text-to-image generation with a 1.25 B-parameter UNet diffusion model, held-out denoising-loss and CLIP-scored evaluation, and bounded LoRA fine-tuning to a set of captioned photographs",
    "intro": (
        "Kandinsky 2.2 (Shakhmatov et al., 2023) is a two-stage latent diffusion architecture: a frozen diffusion prior with "
        "CLIP ViT-G/14 encoders maps text prompts to image embeddings, a 1.25 B-parameter UNet denoises a 4-channel latent "
        "conditioned on those image embeddings with classifier-free guidance, and a MoVQ decoder (67.8 M parameters) decodes "
        "the latent to 3-channel RGB pixels. Three pinned snapshots make one generator: the decoder (5.28 GB safetensors), the "
        "shared diffusion prior (10.57 GB safetensors) and, for evaluation only, a CLIP ViT-B/32 scorer.\n\n"
        "Two architectural properties are handled in the open. **The prior pipeline can be released before training**: "
        "Section 5 encodes every prompt the notebook will ever use — the 60 captions, the generation prompts, the empty "
        "negative prompt — into image embeddings on the CPU, and releases the prior so the UNet, the MoVQ, the scorer and a "
        "training graph fit comfortably on a 16 GB GPU. **Generation has no ground truth**, so the notebook reads three kinds of "
        "number: the held-out *denoising loss* (the training objective, measured on photographs the model never trained on), "
        "CLIP scores of generated images (prompt alignment and similarity to real photographs), and the same CLIP scores on "
        "the real photographs themselves — the ceiling. None of these is a human judgement of image quality."
    ),
    "learning_objectives": (
        "by the end of this notebook you will be able to —\n\n"
        "1. **Explain** how a prompt becomes an image in Kandinsky 2.2 (prior → image embedding → UNet denoising under "
        "classifier-free guidance → MoVQ decoding) and why the notebook can release the prior before training (Section 5).\n"
        "2. **Diagnose** an invalid training record from a validation refusal, and **explain** why the sample split is drawn "
        "per species (Section 4).\n"
        "3. **Interpret** a held-out denoising loss, a CLIP prompt similarity, a zero-shot label accuracy and a reference "
        "similarity, and **compare** the frozen model with the real-photo ceiling (Section 6).\n"
        "4. **Apply** a bounded LoRA fine-tuning with explicit hyperparameters and **identify** which parameters it trains and "
        "how the kept epoch is chosen (Section 7).\n"
        "5. **Compare** the adapted and the frozen model on identical held-out inputs, and **distinguish** what the procedure "
        "guarantees from what is only observed (Section 8).\n"
        "6. **Verify** that an exported adapter, reloaded from files, reproduces the held-out loss and the image of the "
        "in-memory model (Section 9).\n"
        "7. **Predict**, run and **explain** the effect of changing one generation setting, the guidance scale, in an optional "
        "activity (Section 10).\n"
        "8. **Write** an evidence-based conclusion that names the baseline, the ceiling and the limits of twelve held-out "
        "photographs (Conclusion)."
    ),
    "exclusions": (
        "Kandinsky 2.1, inpainting, ControlNet depth conditioning (each in their own repository), full fine-tuning, "
        "DreamBooth identifiers, safety filtering of prompts or images, human preference benchmarks, prompt engineering, "
        "and any claim that a CLIP score or a denoising loss measures image quality. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported **GPU** runtime (Google Colab T4 or better, or a Jupyter kernel with a CUDA GPU of at least 15 GB and Python 3.12). The prior runs in float16 while it encodes prompts and is then released; the UNet runs in float16 (2.5 GB) with the LoRA parameters in float32; the MoVQ decoder runs in float16 (0.13 GB). CPU-only runtimes are not supported for this notebook. About 20 GB of disk is needed for the snapshots.",
        "- **Knowledge:** what a two-stage latent diffusion model does at inference (prompt → image embedding via prior → latent via UNet → pixels via MoVQ), what classifier-free guidance is, what a LoRA adapter changes and what it does not, and why a training loss is not a quality score.",
        "- **Weights:** the decoder, the diffusion prior and the CLIP scorer are all safetensors; nothing is unpickled and no Hub-hosted code is executed — the model classes come from `diffusers`, `transformers` and `peft` on PyPI. Upstream weights are released under Apache-2.0; the scorer is MIT.",
        "- **Data contract:** a record is `{{id, image, caption}}` — an RGB image with shorter side 256..4096 px (resized so the shorter side is 512 px and centre-cropped to 512 × 512; the crop is reported) and a caption of 1..1000 characters. Validation is structural: nothing checks that a caption describes its image or that the model can render it.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — photographs of identifiable people, licensed stock images or client material are exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the Hub, the default path fetches 60 pinned photographs (about 6 MB) from the public iNaturalist open-data bucket `inaturalist-open-data.s3.amazonaws.com` over HTTPS, digest-verified before decoding; every photo is CC0 and its observation page is recorded.",
    ],
    "guided": {
        "engineering_tag": "[Engineering]",
        "opening": [
            (
                "## How to use this notebook\n\n"
                "**Who this notebook is for.** Learners who can open a hosted notebook (Google Colab or Jupyter), run cells in "
                "order and read short Python, and who want to see how a text-to-image diffusion model is evaluated and adapted. "
                "No prior experience with diffusion models is assumed: each term is explained where it is first needed, and the "
                "glossary below collects them. You need a CUDA **GPU** with at least 15 GB of memory — a Colab T4 is enough — and "
                "about 20 GB of free disk; CPU-only runtimes are not supported. The **Prerequisites** give the details.\n\n"
                "**Running it.** In Colab, choose *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*. The default "
                "path needs no edit, no upload, no account and no token. Sections 1–3 install pinned packages and download about "
                "16.5 GB of verified weights, so they take the longest before any model runs; read ahead while they finish, or "
                "run the notebook one cell at a time with *Shift + Enter*.\n\n"
                "**Two kinds of cell.** *Learner cells* (Sections 4–10) are the machine-learning workflow; each prints compact "
                "dictionaries for you to read. *Infrastructure cells* (Sections 1–3: the package install, the carried package "
                "modules and the pinned-weight staging) are collapsed and titled **Infrastructure**. You may run them without "
                "studying their implementation: they exist for reproducibility and provenance, not as prerequisite machine-"
                "learning knowledge. Open one with *Show code* if you are curious.\n\n"
                "**Form controls.** Some learner cells start with fields that Colab renders as a form: `USE_BYOD` (Section 4); "
                "`STEPS`, `GUIDANCE_SCALE` and `IMAGES_PER_PROMPT` (Section 6); `EPOCHS`, `LEARNING_RATE` and `BATCH_SIZE` "
                "(Section 7); and `RUN_ACTIVITY` and `ACTIVITY_GUIDANCE_SCALE` in the optional activity (Section 10). Leave "
                "them at their defaults for the first run: the notes and sample answers describe the default path. Changing a "
                "field and re-running from that cell is how you experiment afterwards.\n\n"
                "**Section tags.** Each numbered heading carries one tag. **[Concept]** — what the model does and why. "
                "**[Evaluation practice]** — how the evidence is produced and how to read it. **[Engineering]** — "
                "reproducibility, provenance and packaging.\n\n"
                "**Predict, then check.** Before each principal result a **Predict before running** prompt asks you to commit to "
                "an expectation; after it, **What to notice** describes normal output and a collapsed **Check your reasoning** "
                "answer follows each checkpoint. Write your own answer first, then open it. Exact numbers vary with the runtime "
                "and the library versions, so the notes describe the shape of a normal result rather than fixed values."
            ),
            (
                "## The task: Input → Model/System → Output\n\n"
                "| Stage | Input | Model / system | Output |\n"
                "|---|---|---|---|\n"
                "| **Generation** | a text prompt | diffusion prior (text → CLIP image embedding) → UNet, with its LoRA adapter, "
                "denoising a random 4 × 64 × 64 latent over `STEPS` steps under classifier-free guidance → MoVQ decoder | one "
                "512 × 512 RGB image |\n"
                "| **Adaptation** | a captioned photograph | MoVQ encoder → latent + seeded noise at a random timestep → UNet "
                "predicts the noise; the mean-squared error to the true noise updates only the LoRA tensors | a small "
                "safetensors adapter |\n"
                "| **Evaluation** | held-out photographs and fixed prompt/seed pairs | the frozen and the adapted model, scored "
                "the same way; a frozen CLIP ViT-B/32 scores the images | a denoising loss per timestep; CLIP scores of "
                "generated images beside the same scores on the real photographs (the ceiling) |\n\n"
                "No stage judges image quality: the loss is the training objective and the CLIP scores are another frozen "
                "model's opinion.\n\n"
                "## Roadmap\n\n"
                "| Section | Tag | What happens | What you read |\n"
                "|---|---|---|---|\n"
                "| 1. Install the pinned runtime | [Engineering] | exact package versions | versions, `cuda: True` |\n"
                "| 2. Pipeline code | [Engineering] | the repository's three modules, carried verbatim | nothing runs yet |\n"
                "| 3. Pin, stage and verify | [Engineering] | three snapshots downloaded and digest-checked | file counts, device |\n"
                "| 4. Sample photographs | [Evaluation practice] | 60 photographs validated and split 36 / 12 / 12 | split sizes, refusals |\n"
                "| 5. Encode prompts, release the prior | [Concept] | prompts → image embeddings, prior dropped | GPU memory |\n"
                "| 6. The frozen model | [Evaluation practice] | baseline loss and CLIP scores, real-photo ceiling | the baseline |\n"
                "| 7. LoRA fine-tuning | [Concept] | a bounded adaptation of the UNet | the epoch history |\n"
                "| 8. Held-out comparison | [Evaluation practice] | frozen vs adapted on identical inputs | the principal result |\n"
                "| 9. New prompt, export, reload | [Engineering] | an unseen prompt; adapter saved and reloaded | reload parity |\n"
                "| 10. Optional activity | [Concept] | change the guidance scale (off by default) | your comparison |\n"
                "| Troubleshooting | [Engineering] | common hosted-runtime failures | when something fails |\n"
                "| Interpretation and conclusion | [Evaluation practice] | limits and an evidence-based conclusion | your conclusion |\n\n"
                "**Fast path.** Short on time? Run all, then read Sections 6 and 8 and the conclusion: they carry the principal "
                "results. The canonical path ends with Section 9; Section 10 changes nothing unless you switch it on."
            ),
            (
                "<details>\n"
                "<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>\n\n"
                "| Term | Meaning in this notebook |\n"
                "|---|---|\n"
                "| **Latent diffusion** | Generating an image by starting from random noise in a compressed *latent* space and removing the noise step by step; a decoder then turns the latent into pixels. |\n"
                "| **Latent** | The compressed representation the UNet works on: 4 channels of 64 × 64 for a 512 × 512 image. |\n"
                "| **Diffusion prior** | Kandinsky's first stage: a model that turns a text prompt into a predicted CLIP *image* embedding. Frozen here. |\n"
                "| **Image embedding** | A vector summarising an image's content; the UNet is conditioned on it rather than on text. |\n"
                "| **UNet** | The 1.25 B-parameter network that predicts the noise in a latent at a given timestep. |\n"
                "| **MoVQ** | The decoder (and encoder) between latents and pixels. Frozen here. |\n"
                "| **Timestep** | How far along the noise schedule a latent is: 0 is almost clean, 999 almost pure noise. |\n"
                "| **Denoising loss** | Mean-squared error between the noise the UNet predicts and the noise that was added. It is the training objective, not a quality score. |\n"
                "| **Classifier-free guidance** | At each generation step the UNet runs with the prompt's embedding and with the empty (negative) prompt's; the difference is amplified by the *guidance scale*. |\n"
                "| **Guidance scale** | How strongly generation follows the prompt. `GUIDANCE_SCALE = 4.0` by default; 1.0 switches guidance off. |\n"
                "| **Seed** | The number that fixes the random noise, so the same prompt, seed and settings give the same image. |\n"
                "| **Frozen model** | The pretrained model with its weights unchanged; here, the UNet with a LoRA adapter whose output starts at zero. |\n"
                "| **LoRA** | Low-rank adaptation: small trainable matrices added to the attention projections, trained while the original weights stay fixed. |\n"
                "| **Rank** | The inner size of the LoRA matrices (8 here); it bounds how much the adapter can change. |\n"
                "| **Epoch** | One pass over the 36 training photographs. |\n"
                "| **Validation / test split** | Validation photographs choose the kept epoch; test photographs are used only for the final comparison. |\n"
                "| **CLIP score** | Cosine similarity between CLIP embeddings of an image and a text (or another image), × 100. |\n"
                "| **Label accuracy** | The share of generated images whose nearest caption, according to CLIP, is their own prompt (zero-shot, argmax rule). |\n"
                "| **Reference similarity** | CLIP similarity between a generated image and the average of the held-out real photographs with the same caption. |\n"
                "| **Real-photo ceiling** | The same CLIP scores computed on the real test photographs: roughly the best a generator could reach on these measures. |\n"
                "| **Adapter / artifact** | The trained LoRA tensors saved as `adapter.safetensors` with a `manifest.json`. |\n"
                "| **Digest (SHA-256)** | A fingerprint of a file's bytes; a single changed byte changes it. |\n"
                "| **BYOD** | Bring Your Own Data: an optional switch to run the same workflow on your own captioned photographs. |\n\n"
                "</details>"
            ),
        ],
        "infrastructure": {
            "install": {
                "title": "install the pinned runtime and print the versions",
                "note": "The next code cell is collapsed. It installs exact package versions and prints them; you may run it without studying its implementation. The learning activities start in Section 4.",
            },
            "modules": {
                "note": "The code cells in this section are collapsed. They are the repository's package, carried so that this notebook works on its own; you may run them without reading them, and nothing in them runs a model.",
            },
            "model": {
                "title": "stage, digest-verify and load the three pinned snapshots",
                "note": "The next code cell is collapsed. It downloads about 16.5 GB of pinned weights, checks every file's size and SHA-256, and loads the model; you may run it without studying its implementation.",
            },
        },
        "after": {
            "install": (
                "**Expected result:** one dictionary with the notebook's source revision, the Python version, the `torch`, "
                "`diffusers`, `transformers` and `peft` versions, and `'cuda': True`. If `'cuda'` is `False`, stop here and see "
                "**Troubleshooting**: this notebook is not supported on a CPU-only runtime."
            ),
            "model": (
                "**What to notice:** the decoder's id, revision, licence and file count; for each of the three snapshots a "
                "`fetched` list of the files downloaded on this run (empty on a rerun, because staging only fetches files that "
                "are absent) and a count of verified files; then `'device': 'cuda'`. A size or SHA-256 mismatch stops the cell "
                "with a `ValueError` naming the file — see **Troubleshooting**, and never edit a manifest to get past one."
            ),
        },
    },
    "cells": [
        {
            "md": (
                "## 4. Sample photographs, validation and splits · [Evaluation practice]\n\n"
                "The default dataset is 60 research-grade iNaturalist photographs of six common North American birds — 10 per "
                "species, one per observer per species, every one CC0 — fetched by photo id from the open-data bucket and "
                "refused on any byte-size or SHA-256 mismatch (`fetch_corpus`). Each photo's caption is generated from its "
                "species by one template, so the adaptation teaches the generator what six names look like in this kind of "
                "photograph. `build_sample_dataset` draws a seeded stratified split — 6 / 2 / 2 per species for training, "
                "validation and test — and `dataset_manifest` validates every split, checks that no image appears twice and "
                "records a digest.\n\n"
                "A **caption** is the text paired with a photograph; here it doubles as the prompt the model trains and generates "
                "with. **Validation** checks each record's structure before any model sees it; a **refusal probe** is a "
                "deliberately broken record used to show that the check works.\n\n"
                "**Expected result:** 36 / 12 / 12 records, six distinct captions, a shorter side around 300..500 px (every photo is "
                "centre-cropped to 512²), a written `outputs/{stem}_sample_captions.csv` in the shape BYOD expects, and "
                "three refusal probes — a missing caption, a 200 px image, a duplicate id — each rejected before the model runs."
            ),
            "code": (
                "import json\n"
                "import os\n"
                "from pathlib import Path\n\n"
                "import numpy as np\n"
                "from PIL import Image\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_path = Path('work') / file_name\n"
                "    byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_path.write_bytes(payload)\n"
                "    splits = split_dataset(load_byod_dataset(byod_path), seed=0)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "else:\n"
                "    splits = fetch_sample_dataset(cache_dir='weights/inat-birds')\n"
                "    data_source = SAMPLE_LABEL_SOURCE\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n\n"
                "dataset_report = dataset_manifest({{'train': train_records, 'validation': val_records, 'test': test_records}})\n"
                "print({{'data_source': data_source, 'splits': {{k: v['n_records'] for k, v in dataset_report['splits'].items()}}, 'captions': dataset_report['splits']['train']['n_captions'], 'disjoint': dataset_report['disjoint']}})\n"
                "print({{'shorter_side': dataset_report['splits']['train']['shorter_side'], 'centre_cropped': dataset_report['splits']['train']['centre_cropped'], 'digest': dataset_report['digest'][:16] + '...'}})\n"
                "print({{'first_test_record': validate_inputs(test_records[0]), 'caption': test_records[0]['caption']}})\n"
                "prompts = sample_prompts(train_records)\n"
                "print({{'prompts': prompts}})\n"
                "sample_csv = write_dataset_csv(test_records, 'outputs/{stem}_sample_captions.csv')\n"
                "print({{'sample_csv': str(sample_csv)}})\n\n"
                "print({{'validation': INPUT_SCHEMA['validation']}})\n"
                "probes = {{\n"
                "    'missing caption': [{{'id': r['id'], 'image': r['image']}} for r in train_records[:4]],\n"
                "    'image too small': [{{**train_records[0], 'image': Image.new('RGB', (200, 200))}}, *train_records[1:4]],\n"
                "    'duplicate id': [train_records[0], *train_records[:4]],\n"
                "}}\n"
                "for name, records in probes.items():\n"
                "    try:\n"
                "        validate_dataset(records)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "**What to notice:** `data_source` names the sample, `disjoint` is `True`, and each refusal probe prints "
                "`rejected` with a message that names the broken record and the rule it broke — the same messages you would "
                "see for a bad BYOD record. With `USE_BYOD = True` the split sizes come from your own captions instead.\n\n"
                "**Checkpoint:** why does the sample split draw 6 / 2 / 2 photographs *per species* instead of shuffling all 60 "
                "together, and what would go wrong in Section 8 if it did not?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "Each species has one caption, so a per-species (stratified) split guarantees that every caption appears in "
                "training, in validation and in test in the same proportion. A plain shuffle of 60 photographs could leave a "
                "species with no test photograph, or with none to train on; the held-out comparison would then cover a "
                "different mix of species than the training did, and a change in the average could come from the mix rather "
                "than from the adaptation. The corpus also takes one photograph per observer per species, so one "
                "photographer's near-duplicate shots cannot sit on both sides of the split. The refusals happen before any "
                "model runs, so a malformed record cannot cost GPU time or enter training silently — but validation is "
                "structural only: it cannot tell whether a caption describes its photograph.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 5. Encode every prompt with the Prior, then release it · [Concept]\n\n"
                "`pipe.encode_prompts` loads the Kandinsky 2.2 diffusion prior from the verified snapshot (float16 on CUDA), "
                "runs the prior diffusion process to generate predicted CLIP image embeddings for each distinct prompt, and "
                "keeps the embeddings on the CPU. Encoded here: the six training captions (which are also the validation and "
                "test captions and the generation prompts), one new prompt for Section 9, and the empty negative prompt that "
                "classifier-free guidance needs. `release_prior` then drops the 1.03 B parameter prior so the UNet, the MoVQ, "
                "the scorer and a training graph fit on a 16 GB GPU.\n\n"
                "**Predict before running:** after `release_prior`, will GPU memory fall by a little or by most of what the "
                "prior occupied? And once the prior is gone, which prompts can this notebook still generate without loading "
                "it again?\n\n"
                "**Expected result:** the prior loading from the verified safetensors, seven or eight prompts encoded in seconds, "
                "`prior_released: True`, and GPU memory falling back after the release."
            ),
            "code": (
                "import time\n\n"
                "NEW_PROMPT = 'a photo of a House Finch (Haemorhous mexicanus) perched on a snow-covered branch in winter'\n\n"
                "def gpu_memory_gb():\n"
                "    return round(torch.cuda.memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None\n\n"
                "all_prompts = sample_prompts(train_records + val_records + test_records) + [NEW_PROMPT]\n"
                "encode_report = pipe.encode_prompts(all_prompts)\n"
                "print({{**encode_report, 'gpu_memory_gb_with_prior': gpu_memory_gb()}})\n"
                "released = pipe.release_prior()\n"
                "print({{'prior_released': released, 'gpu_memory_gb_after_release': gpu_memory_gb(), 'device': pipe.device, 'precision': str(pipe.dtype).replace('torch.', '')}})"
            ),
        },
        {
            "md": (
                "**What to notice:** `encoded` counts the distinct prompts (the empty negative prompt included), "
                "`prior_released` is `True`, and `gpu_memory_gb_after_release` is well below `gpu_memory_gb_with_prior`. The "
                "exact figures depend on the GPU and the library versions.\n\n"
                "**Checkpoint:** why can every prompt be encoded *before* training, and never need the prior again?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "The prior is frozen: neither training nor generation changes it, and each prompt is encoded with a seed "
                "derived from the prompt text, so its embedding is fixed. The UNet is conditioned only on that image "
                "embedding, so an embedding computed once serves the frozen UNet, the adapted UNet and the reloaded one "
                "alike. The price is that only prompts encoded here are cheap later: `generate` loads the prior again for a "
                "prompt that is not in the cache. That is why Section 5 also encodes the Section 9 prompt, and why the optional "
                "activity in Section 10 reuses the same prompts.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 6. The frozen model: held-out denoising loss and CLIP-scored generations · [Evaluation practice]\n\n"
                "**Question tested:** before any training, how well does the pretrained model fit these photographs, and how "
                "far are its generations from the real-photo ceiling? These numbers are the **baseline** for Section 8.\n\n"
                "The pipeline was built with `use_lora=True`: the adapter's B matrices start at zero, so until Section 7 this is "
                "the pretrained model. Two kinds of number are read here and kept for the comparison.\n\n"
                "**Held-out denoising loss** (`pipe.evaluate`): each held-out photograph is MoVQ-encoded, noised at five fixed "
                "timesteps (100, 300, 500, 700, 900) with a seeded noise tensor, and the UNet's noise prediction is scored "
                "against that noise (MSE over the latent). It is the training objective measured on photographs the model never "
                "trains on; the same seed gives the same latents, noise and timesteps later, so the adapted number is a paired "
                "comparison, not a re-draw.\n\n"
                "**CLIP-scored generations** (`pipe.generate` + `score_generations`): two images per training caption at fixed "
                "seeds (20 steps, guidance 4.0), scored by the frozen CLIP ViT-B/32 on prompt alignment (cosine × 100), "
                "zero-shot label accuracy (which of the six captions is nearest) and similarity to the mean embedding of the "
                "held-out real photographs. `real_photo_baseline` scores the real test photographs the same way: the ceiling.\n\n"
                "**Predict before running:** at which of the five timesteps do you expect the denoising loss to be highest? And "
                "which CLIP measure — prompt similarity, label accuracy or reference similarity — do you expect to sit furthest "
                "below the real-photo ceiling for the frozen model?"
            ),
            "code": (
                "STEPS = 20  # @param {{type:\"integer\"}}\n"
                "GUIDANCE_SCALE = 4.0  # @param {{type:\"number\"}}\n"
                "IMAGES_PER_PROMPT = 2  # @param {{type:\"integer\"}}\n"
                "EVAL_SEED = 0\n\n"
                "def grid(images, path, columns=6):\n"
                "    tiles = [im.resize((256, 256)) for im in images]\n"
                "    rows = (len(tiles) + columns - 1) // columns\n"
                "    sheet = Image.new('RGB', (256 * columns, 256 * rows), 'white')\n"
                "    for i, tile in enumerate(tiles):\n"
                "        sheet.paste(tile, (256 * (i % columns), 256 * (i // columns)))\n"
                "    sheet.save(path)\n"
                "    return path\n\n"
                "scorer = ClipScorer(weights_dir=SCORER_WEIGHTS_DIR, device=pipe.device)\n"
                "t0 = time.perf_counter()\n"
                "frozen_val = pipe.evaluate(val_records, seed=EVAL_SEED)\n"
                "frozen_test = pipe.evaluate(test_records, seed=EVAL_SEED)\n"
                "print({{'frozen_denoising_mse': {{'validation': frozen_val['denoising_mse'], 'test': frozen_test['denoising_mse']}}, 'by_timestep_test': frozen_test['by_timestep'], 'seconds': round(time.perf_counter() - t0, 1)}})\n\n"
                "generation_prompts = [p for p in prompts for _ in range(IMAGES_PER_PROMPT)]\n"
                "t0 = time.perf_counter()\n"
                "frozen_generation = pipe.generate(generation_prompts, seed=1000, steps=STEPS, guidance_scale=GUIDANCE_SCALE)\n"
                "print({{'generated': len(frozen_generation['images']), 'steps': frozen_generation['steps'], 'guidance_scale': frozen_generation['guidance_scale'], 'seconds': frozen_generation['seconds'], 'adapted': frozen_generation['model']['adapted']}})\n"
                "frozen_scores = score_generations(scorer, frozen_generation['images'], references=test_records)\n"
                "real_ceiling = real_photo_baseline(scorer, test_records)\n"
                "print({{'frozen_generations': {{k: frozen_scores[k] for k in ('clip_prompt_similarity', 'label_accuracy', 'reference_similarity')}}}})\n"
                "print({{'real_photo_ceiling': {{k: real_ceiling[k] for k in ('clip_prompt_similarity', 'label_accuracy', 'reference_similarity')}}}})\n"
                "for entry in frozen_scores['per_image'][::IMAGES_PER_PROMPT]:\n"
                "    print({{'prompt': entry['prompt'][:42], 'clip': entry['clip_prompt_similarity'], 'nearest': entry['nearest_prompt'][13:40], 'correct': entry['correct'], 'reference_similarity': entry['reference_similarity']}})\n"
                "print({{'grid': str(grid([g['image'] for g in frozen_generation['images']], 'outputs/{stem}_frozen_grid.jpg'))}})"
            ),
        },
        {
            "md": (
                "**What to notice:** a validation and a test denoising loss, the test loss broken down by timestep, then twelve "
                "generated images with three CLIP measures beside the same three measures on the real photographs. One line per "
                "caption shows the prompt, its CLIP score, the caption CLIP found nearest and whether that was the right one. "
                "`outputs/{stem}_frozen_grid.jpg` holds the twelve images: open it from the file browser and compare what you see "
                "with the numbers.\n\n"
                "**Checkpoint:** read `by_timestep_test`. Where is the loss highest, and why is a denoising loss not a measure "
                "of how good the generated images look?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "The UNet predicts the added noise. At a large timestep the noisy latent is mostly noise, so the noise is "
                "comparatively easy to recover; at a small timestep the noise is a faint perturbation of the image and is harder "
                "to separate out. The loss is therefore usually highest at the smallest timesteps — but read your own "
                "`by_timestep_test` rather than assuming it. The loss measures how well the UNet fits *these photographs* under "
                "the training objective. It says nothing directly about composition, sharpness or whether a person would find "
                "an image convincing, which is why the notebook also reads CLIP scores and keeps the real-photo ceiling beside "
                "them. The CLIP scores are another model's opinion, not a human judgement: a generated image can score close "
                "to the ceiling and still look wrong to a birdwatcher. With twelve images, label accuracy moves in steps of "
                "1/12, so one image changes it by about 0.08.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 7. Bounded LoRA fine-tuning · [Concept]\n\n"
                "`pipe.adapt` trains the 176 LoRA tensors (rank 8, 1,646,592 parameters — 0.13 % of the UNet) that "
                "`peft` attached to the attention projections of the UNet, and nothing else; the UNet base, the MoVQ "
                "and the prior are frozen. Each step takes one training photograph's latent (MoVQ-encoded once, seeded), "
                "draws a timestep uniformly from the 1,000-step schedule and a noise tensor (both seeded), adds the noise, "
                "and minimises the MSE between the predicted and the true noise; AdamW at a fixed learning rate, gradient-norm "
                "clipping at 1.0, float16 autocast with loss scaling on CUDA. Epoch 0 records the frozen model's validation loss, "
                "and the epoch with the lowest validation denoising loss is kept.\n\n"
                "**Predict before running:** will the training loss fall smoothly from epoch to epoch? Will the kept (best) epoch "
                "be the last one?"
            ),
            "code": (
                "EPOCHS = 4  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-4  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 1  # @param {{type:\"integer\"}}\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4), 'val_denoising_mse': entry['val_loss']}}\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, seed=EVAL_SEED, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['adapter']['n_trainable'], 'total_parameters': adapt_result['adapter']['n_total'], 'steps': adapt_result['steps'], 'best_epoch': adapt_result['best_epoch'], 'precision': adapt_result['adapter']['precision'], 'seconds': adapt_seconds, 'gpu_memory_gb': gpu_memory_gb()}})"
            ),
        },
        {
            "md": (
                "**What to notice:** one line per epoch — epoch 0 is the frozen model's validation loss with no training loss — "
                "then a summary with 1,646,592 trainable parameters out of the UNet's total, the number of optimiser steps, the "
                "`best_epoch`, the precision and the elapsed seconds.\n\n"
                "**Checkpoint:** why is the kept epoch chosen on the validation photographs and not on the test photographs?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "Choosing the epoch is a decision made by looking at scores. If it were made on the test photographs, the test "
                "result would be the best of several tries on the same data and would overstate how the adapter does on "
                "photographs it has never influenced. Keeping the test photographs out of every decision makes Section 8 a "
                "held-out measurement. The training loss is noisy because each step draws a batch (one photograph by default), a random "
                "timestep and random noise, and the loss depends strongly on the timestep; a single epoch's value is not a trend. "
                "Because epoch 0 (the frozen model) is a candidate, the kept validation loss can never be worse than the "
                "frozen one — if no epoch improves on it, epoch 0 is kept and the adapter stays at its zero initialisation.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation: the paired comparison · [Evaluation practice]\n\n"
                "**Question tested:** on test photographs and prompt/seed pairs that played no part in training or epoch "
                "selection, does the adapted model fit the photographs better than the frozen one, and do its generations move "
                "towards the real-photo ceiling?\n\n"
                "The test photographs were never used for training or epoch selection. The adapted model is scored exactly as "
                "the frozen model was in Section 6 — the same seed, so the same latents, noise and timesteps, and the same "
                "twelve prompt/seed pairs for generation — and the table puts the frozen, the adapted and the real-photo "
                "numbers side by side. The cell asserts only what the procedure guarantees — the kept epoch's validation loss is no "
                "higher than the frozen model's (epoch 0) and the re-scored validation loss matches the history.\n\n"
                "**Predict before running:** will the test denoising loss fall, and by a lot or a little? Which CLIP measure do "
                "you expect to move most after adaptation, and will any of them reach the real-photo ceiling?"
            ),
            "code": (
                "adapted_val = pipe.evaluate(val_records, seed=EVAL_SEED)\n"
                "adapted_test = pipe.evaluate(test_records, seed=EVAL_SEED)\n"
                "adapted_generation = pipe.generate(generation_prompts, seed=1000, steps=STEPS, guidance_scale=GUIDANCE_SCALE)\n"
                "adapted_scores = score_generations(scorer, adapted_generation['images'], references=test_records)\n"
                "comparison = {{\n"
                "    'denoising_mse_validation': {{'frozen': frozen_val['denoising_mse'], 'adapted': adapted_val['denoising_mse']}},\n"
                "    'denoising_mse_test': {{'frozen': frozen_test['denoising_mse'], 'adapted': adapted_test['denoising_mse']}},\n"
                "    'denoising_mse_test_by_timestep': {{t: {{'frozen': frozen_test['by_timestep'][t], 'adapted': adapted_test['by_timestep'][t]}} for t in adapted_test['by_timestep']}},\n"
                "    'clip_prompt_similarity': {{'frozen': frozen_scores['clip_prompt_similarity'], 'adapted': adapted_scores['clip_prompt_similarity'], 'real_photos': real_ceiling['clip_prompt_similarity']}},\n"
                "    'label_accuracy': {{'frozen': frozen_scores['label_accuracy'], 'adapted': adapted_scores['label_accuracy'], 'real_photos': real_ceiling['label_accuracy']}},\n"
                "    'reference_similarity': {{'frozen': frozen_scores['reference_similarity'], 'adapted': adapted_scores['reference_similarity'], 'real_photos': real_ceiling['reference_similarity']}},\n"
                "}}\n"
                "for name, row in comparison.items():\n"
                "    print({{name: row}})\n"
                "for before, after in zip(frozen_scores['per_image'][::IMAGES_PER_PROMPT], adapted_scores['per_image'][::IMAGES_PER_PROMPT]):\n"
                "    print({{'prompt': before['prompt'][:42], 'reference_similarity': {{'frozen': before['reference_similarity'], 'adapted': after['reference_similarity']}}, 'correct': {{'frozen': before['correct'], 'adapted': after['correct']}}}})\n"
                "print({{'grid': str(grid([g['image'] for g in adapted_generation['images']], 'outputs/{stem}_adapted_grid.jpg'))}})\n"
                "evaluation_report = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'components': {{'id': PRIOR_ID, 'revision': PRIOR_REVISION}},\n"
                "    'scorer': frozen_scores['scorer'],\n"
                "    'data_source': data_source,\n"
                "    'dataset': dataset_report,\n"
                "    'generation': {{'steps': STEPS, 'guidance_scale': GUIDANCE_SCALE, 'images_per_prompt': IMAGES_PER_PROMPT, 'seed': 1000}},\n"
                "    'frozen': {{'validation': frozen_val, 'test': frozen_test, 'generations': frozen_scores}},\n"
                "    'adapted': {{'validation': adapted_val, 'test': adapted_test, 'generations': adapted_scores}},\n"
                "    'real_photo_ceiling': real_ceiling,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report, f, indent=2)\n"
                "best = adapt_result['history'][adapt_result['best_epoch']]\n"
                "assert best['val_loss'] <= adapt_result['history'][0]['val_loss']\n"
                "assert abs(adapted_val['denoising_mse'] - best['val_loss']) < 1e-4\n"
                "print({{'test_denoising_mse_change': round(adapted_test['denoising_mse'] - frozen_test['denoising_mse'], 6), 'note': 'held-out observation, not asserted'}})\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "**What to notice:** each row of the comparison pairs a frozen and an adapted value measured on identical "
                "inputs, with the real photographs' value beside the CLIP rows. `test_denoising_mse_change` is printed as a "
                "held-out observation, not asserted: a negative value means the adapted model fits the unseen test photographs "
                "better. Compare `outputs/{stem}_adapted_grid.jpg` with the frozen grid from Section 6 — same prompts, same "
                "seeds, so any difference comes from the adapter.\n\n"
                "**Checkpoint:** which of these results does the procedure *guarantee*, and which are *observations* that could "
                "come out differently on another run or another dataset?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "Guaranteed: the kept epoch's validation loss is no higher than the frozen model's, because epoch 0 is one of "
                "the candidates — the cell asserts exactly that and nothing more. Everything on the test photographs is an "
                "observation: the test loss can rise even when validation fell, and the CLIP measures can move in either "
                "direction. The comparison is *paired* — same latents, noise, timesteps, prompts and seeds — so a difference is "
                "not a re-draw of randomness; but it is still one seeded run on 12 photographs and 12 images, with no estimate "
                "of spread. A small change in label accuracy is one or two images. A defensible sentence has the form: \"on "
                "these 12 held-out photographs, the adapter changed the test denoising loss from *a* to *b* and the reference "
                "similarity from *c* to *d*, against a real-photo ceiling of *e*\". \"The adapter makes better bird pictures\" "
                "is not supported by these numbers.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 9. A new prompt, artifact export and fresh reload · [Engineering]\n\n"
                "The adapted model renders `NEW_PROMPT` — a composition that appears in no training caption — at two seeds; "
                "the CLIP prompt similarity is printed as a sanity check, not an evaluation.\n\n"
                "`pipe.save_artifact` writes the 176 trained tensors (about 6.6 MB) as `adapter.safetensors` with a "
                "`manifest.json` recording the artifact format, the decoder's id and revision, the prior snapshot's id "
                "and revision, the LoRA configuration, the tensor names, the file size and SHA-256, the training configuration "
                "and the epoch history (OUT8). `KandinskyPipeline.from_artifact` re-verifies both snapshots, checks the "
                "manifest, the LoRA scope and the digest **before** deserialising, loads a fresh UNet with the adapter "
                "attached and overlays the tensors — a new object from files, not the in-memory model (VER2). The fresh "
                "pipeline adopts the prompt embeddings already encoded, and the cell asserts that it reproduces the same "
                "held-out denoising loss and the same image for the same prompt and seed (VER4).\n\n"
                "**Expected result:** a CLIP prompt similarity for the new prompt labelled as a sanity check, an artifact of "
                "176 tensors of about 6.6 MB with its SHA-256, a `reload_parity` dictionary whose two differences are at or "
                "near zero, and a listing of every file in `outputs/`."
            ),
            "code": (
                "import platform\n"
                "import shutil\n\n"
                "new_generation = pipe.generate([NEW_PROMPT, NEW_PROMPT], seed=2000, steps=STEPS, guidance_scale=GUIDANCE_SCALE)\n"
                "new_scores = score_generations(scorer, new_generation['images'])\n"
                "print({{'new_prompt': NEW_PROMPT, 'clip_prompt_similarity': new_scores['clip_prompt_similarity'], 'seconds': new_generation['seconds'], 'note': 'sanity check, not an evaluation'}})\n"
                "for i, entry in enumerate(new_generation['images']):\n"
                "    entry['image'].save(f'outputs/{stem}_new_prompt_{{i}}.png')\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': artifact_manifest['weights']['n_tensors'], 'bytes': artifact_manifest['weights']['bytes'], 'sha256': artifact_manifest['weights']['sha256'][:16] + '...'}})\n\n"
                "reloaded = KandinskyPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, prior_dir=PRIOR_WEIGHTS_DIR, device=pipe.device)\n"
                "reloaded.import_prompt_cache(pipe.export_prompt_cache())\n"
                "reloaded_test = reloaded.evaluate(test_records, seed=EVAL_SEED)\n"
                "before = pipe.generate([prompts[0]], seed=3000, steps=STEPS, guidance_scale=GUIDANCE_SCALE)['images'][0]['image']\n"
                "after = reloaded.generate([prompts[0]], seed=3000, steps=STEPS, guidance_scale=GUIDANCE_SCALE)['images'][0]['image']\n"
                "parity = {{'denoising_mse_diff': round(abs(reloaded_test['denoising_mse'] - adapted_test['denoising_mse']), 8), 'mean_abs_pixel_diff': round(float(np.abs(np.asarray(before, dtype=np.float32) - np.asarray(after, dtype=np.float32)).mean()), 4)}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['denoising_mse_diff'] < 1e-6 and parity['mean_abs_pixel_diff'] < 1.0\n\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model': {{**evaluation_report['model'], 'model_license': MODEL_LICENSE, 'device': pipe.device, 'precision': str(pipe.dtype).replace('torch.', ''), 'source': pipe.source}},\n"
                "    'components': {{**evaluation_report['components'], 'license': PRIOR_LICENSE}},\n"
                "    'scorer': {{**evaluation_report['scorer'], 'license': SCORER_LICENSE}},\n"
                "    'provenance': {{\n"
                "        'snapshots': {{'decoder': len(MANIFEST['files']), 'prior': len(PRIOR_MANIFEST['files']), 'scorer': len(SCORER_MANIFEST['files'])}},\n"
                "        'safetensors_only': True,\n"
                "        'remote_code_executed': False,\n"
                "        'prior_released_before_training': released,\n"
                "        'data_base_url': CORPUS_BASE_URL,\n"
                "        'data_license': CORPUS_LICENSE,\n"
                "    }},\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'diffusers': diffusers.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__}},\n"
                "    'data_source': data_source,\n"
                "    'comparison': comparison,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['weights']['sha256'], 'bytes': artifact_manifest['weights']['bytes']}},\n"
                "    'reload_parity': parity,\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(result_payload, f, indent=2)\n\n"
                "print('outputs/:')\n"
                "for path in sorted(Path('outputs').rglob('*')):\n"
                "    if path.is_file():\n"
                "        print(f'  - {{path.as_posix()}} ({{path.stat().st_size / 1024:.1f}} KB)')"
            ),
        },
        {
            "md": (
                "**What to notice:** the reloaded pipeline is a new object built from `adapter.safetensors`, `manifest.json` "
                "and the verified base snapshots, not the model in memory — so matching numbers show that the files alone carry "
                "the adaptation. Reload parity shows that a saved computation can be reproduced; it says nothing about whether "
                "the adapter is good. The new prompt's CLIP score is one composition at two seeds: a sanity check, not a test "
                "of generality. This is the end of the canonical path; everything it produces is in `outputs/`."
            ),
        },
        {
            "md": (
                "## 10. Optional activity: change one thing — the guidance scale · [Concept]\n\n"
                "**Predict → Change one thing → Run → Observe → Explain.** This activity is off by default and changes nothing "
                "the canonical path produced: with `RUN_ACTIVITY = False` the next cell only prints how to switch it on. It "
                "reuses the adapted model, the twelve prompt/seed pairs of Section 8 and the prompt embeddings already encoded, "
                "so it downloads nothing and does not reload the prior.\n\n"
                "**Change one thing:** generation uses classifier-free guidance at `GUIDANCE_SCALE` (4.0 by default). Set "
                "`ACTIVITY_GUIDANCE_SCALE` to another value between 1.0 and 20.0 (outside that range `generate` refuses) and "
                "`RUN_ACTIVITY = True`, then run the cell. Prompts, seeds, steps and the adapter stay fixed; only the guidance "
                "scale changes. At exactly 1.0 the pipeline skips guidance: the UNet sees only the prompt's embedding and the "
                "negative prompt is not used.\n\n"
                "**Predict before running:** with guidance lowered to 1.0, will CLIP prompt similarity rise or fall? What about "
                "label accuracy and reference similarity? Write your prediction down."
            ),
            "code": (
                "RUN_ACTIVITY = False  # @param {{type:\"boolean\"}}\n"
                "ACTIVITY_GUIDANCE_SCALE = 1.0  # @param {{type:\"number\"}}\n\n"
                "if RUN_ACTIVITY:\n"
                "    activity_generation = pipe.generate(generation_prompts, seed=1000, steps=STEPS, guidance_scale=ACTIVITY_GUIDANCE_SCALE)\n"
                "    activity_scores = score_generations(scorer, activity_generation['images'], references=test_records)\n"
                "    for key in ('clip_prompt_similarity', 'label_accuracy', 'reference_similarity'):\n"
                "        print({{key: {{f'guidance {{GUIDANCE_SCALE}}': adapted_scores[key], f'guidance {{ACTIVITY_GUIDANCE_SCALE}}': activity_scores[key], 'real_photos': real_ceiling[key]}}}})\n"
                "    print({{'seconds': {{f'guidance {{GUIDANCE_SCALE}}': adapted_generation['seconds'], f'guidance {{ACTIVITY_GUIDANCE_SCALE}}': activity_generation['seconds']}}}})\n"
                "    print({{'grid': str(grid([g['image'] for g in activity_generation['images']], 'outputs/{stem}_activity_grid.jpg'))}})\n"
                "else:\n"
                "    print({{'activity': 'skipped (optional)', 'to_run': 'set RUN_ACTIVITY = True and a guidance scale in 1.0..20.0, then run this cell'}})"
            ),
        },
        {
            "md": (
                "**Observe:** three rows, each with the value at the default guidance, at your guidance and on the real "
                "photographs, then the generation time at both settings; compare `outputs/{stem}_activity_grid.jpg` with "
                "`outputs/{stem}_adapted_grid.jpg`.\n\n"
                "**Explain:** did the result match your prediction? Which measure moved most, and does what you see in the two "
                "grids agree with the numbers?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "Guidance amplifies the difference between the prompt-conditioned and the unconditioned prediction, so a lower "
                "scale usually follows the prompt less closely — CLIP prompt similarity and label accuracy often fall — while "
                "images can look less saturated and more varied. A higher scale usually pushes the other way, up to a point "
                "where images become over-contrasted. \"Usually\" is the honest word: twelve images at one seed each are a small "
                "sample, and a result against the usual direction is an observation about this run, not a failed activity. "
                "Guidance also costs time: above 1.0 the UNet evaluates both the prompt and the negative prompt at every step, "
                "so compare the two `seconds` values. Because only the guidance scale changed, any difference is caused by it "
                "— the adapter, the prompts and the seeds were held fixed. This is a generation setting, not training: the "
                "held-out denoising loss does not depend on it.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## Troubleshooting · [Engineering]\n\n"
                "| Symptom | Likely cause | What to do |\n"
                "|---|---|---|\n"
                "| Section 1 prints `'cuda': False`, or a later cell reports that CUDA is not available | the runtime has no GPU | "
                "*Runtime → Change runtime type → T4 GPU*, then run all again from the top. This notebook is not supported on a "
                "CPU-only runtime. If Colab offers no GPU, your GPU quota may be exhausted; try again later. |\n"
                "| `RuntimeError: Core dependencies changed while older modules were loaded` | a pinned install replaced a package "
                "this runtime had already imported | *Runtime → Restart session*, then run all again from the top. Do not remove "
                "pins to get past it. |\n"
                "| `pip` exits with an error in Section 1 | a transient network or package-index failure | Re-run the Section 1 "
                "cell; if it keeps failing, restart the session and run all again. |\n"
                "| A download error (timeout, HTTP 429/5xx) in Section 3 | a transient Hugging Face Hub failure | Re-run the Section 3 "
                "cell: staging only fetches the files that are still absent. |\n"
                "| `ValueError: ...: size ... != manifest ...` or `sha256 ... != manifest ...` | a partial or corrupted download "
                "(staging checks that a file exists, verification checks its bytes) | Delete the named file under `weights/` and "
                "re-run the Section 3 cell. Never edit a manifest to get past a mismatch. |\n"
                "| `No space left on device` | the snapshots need about 16.5 GB plus the packages | Start a fresh runtime with about "
                "20 GB free; delete an old `weights/` copy if you have one. |\n"
                "| A photograph fetch fails in Section 4 (`URLError`, or `fetched ... bytes with sha256 ..., pinned ...`) | a "
                "network failure or an unexpected response from the iNaturalist bucket | Re-run the Section 4 cell; photographs "
                "already verified are kept in `weights/inat-birds/` and reused. |\n"
                "| `CUDA out of memory` | another large object is still on the GPU, or a form field was raised (`BATCH_SIZE`, "
                "`IMAGES_PER_PROMPT`) | Restart the session and run all with the default fields. Check that Section 5 printed "
                "`prior_released: True`. |\n"
                "| `ModuleNotFoundError: No module named 'google.colab'` with `USE_BYOD = True` | the upload dialog needs Google Colab | Use Colab for "
                "BYOD, or keep `USE_BYOD = False`. |\n"
                "| `BYOD zip must contain captions.csv` or `captions.csv is missing columns [...]` | the zip layout differs from "
                "the contract | Put `captions.csv` with the columns `id`, `file`, `caption` in the zip beside the images; "
                "`outputs/{stem}_sample_captions.csv` shows the expected shape. |\n"
                "| `KeyError` naming an image file during BYOD loading | a `file` value that is not the bare name of an image in "
                "the zip | Use the file name only (`bird01.jpg`, not `images/bird01.jpg`) and make sure every listed image is "
                "in the zip. |\n"
                "| `image sides must be within 256..4096 px`, `split leaves no test record` or `split leaves ... training records` "
                "| a BYOD image is too small or too large, or there are too few images per caption | Resize the image; give at "
                "least one caption three or more images and provide at least four training images in total. |"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits · [Evaluation practice]\n\n"
        "The question this notebook can answer is narrow: does a LoRA of 1.6 million parameters, trained for a few minutes on "
        "36 captioned photographs, change the held-out denoising loss on twelve photographs the model never saw, and move its "
        "generations towards the held-out real photographs of the same species? Your run answers it in Section 8, on "
        "identical inputs before and after, with the real-photo ceiling beside the CLIP measures; read the direction and the "
        "size of each change there rather than from this text. The artifact that carries the adaptation is about 6.6 MB.\n\n"
        "The numbers are sample-sanity evidence. A denoising loss is the training objective, not a quality score; CLIP "
        "similarity and CLIP's nearest-caption vote are a frozen model's opinion, not a human judgement, and CLIP itself has "
        "biases about what a species name looks like; twelve images per model from one seeded run give no dispersion "
        "estimate; and nothing here measures aesthetics, diversity, artefacts or prompt fidelity beyond the six captions. "
        "Fine-tuning on a narrow domain can also erode the model elsewhere — the new prompt in Section 9 is a sanity check on "
        "one composition, not a test of generality.\n\n"
        "Three things to carry to real data. **Captions are the contract:** the adapter learns the association between the "
        "caption text and the images; a caption that does not describe its image, or one caption for very different images, "
        "teaches noise. **Hold out by caption, not by image:** the split keeps every caption's images across sets so the "
        "held-out loss measures generalisation within the domain; a caption with one image cannot be evaluated. **Licences "
        "travel with the outputs:** the Kandinsky weights are Apache-2.0, the training photographs here are CC0 — with your own "
        "data, the rights to the images and to what the adapter produces are yours to establish.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can stage and digest-verify three pinned safetensors snapshots, fetch and validate digest-pinned real "
        "photographs, encode prompts and release the prior, execute bounded LoRA fine-tuning, evaluate the frozen and the "
        "adapted model on identical held-out inputs with a real-photo ceiling, and emit the shown machine-readable artifacts — "
        "without the repository being reachable. It does **not** establish benchmark superiority, production fitness, or image "
        "quality beyond the checks shown.\n\n"
        "## Conclude with evidence · [Evaluation practice]\n\n"
        "Complete this in your own words, using the numbers your run printed:\n\n"
        "> On [12 held-out photographs of six bird species / your BYOD test set], adapting Kandinsky 2.2 with a rank-8 LoRA "
        "for [epochs] epochs (kept epoch [best_epoch]) changed the test denoising loss from [frozen] to [adapted]. For twelve "
        "images at fixed prompts and seeds, CLIP prompt similarity went from [frozen] to [adapted], label accuracy from "
        "[frozen] to [adapted] and reference similarity from [frozen] to [adapted], against a real-photo ceiling of "
        "[ceiling values]. The most important failure mode or uncertainty is [for example: one seeded run with no spread "
        "estimate; CLIP's own view of a species name; a species the generations still miss]. These numbers do not show "
        "[image quality to a human viewer / behaviour on other prompts or domains / ...]. Next I would [specific next "
        "experiment].\n\n"
        "<details>\n<summary>Check your reasoning: what makes a conclusion strong? (open after writing yours)</summary>\n\n"
        "A strong conclusion names the task and the data (how many held-out photographs, which captions), reports the frozen "
        "baseline and the real-photo ceiling beside the adapted numbers, and keeps the guaranteed result (validation loss no "
        "worse than epoch 0) apart from the observations (everything on the test set). It says that the denoising loss is "
        "the training objective and that CLIP scores are another model's opinion, so neither is a quality judgement. It "
        "states the limits of one seeded run on twelve photographs and does not generalise beyond six species in one "
        "photographic style. It ends with a specific next step — several seeds to estimate spread, a larger held-out set, "
        "or a human rating of the two grids. A weak conclusion says only that \"fine-tuning improved the model\".\n\n"
        "</details>\n\n"
        "**Transfer:** switch on BYOD (Section 4) with your own captioned photographs of one narrow domain — at least three "
        "images per caption — and write the same conclusion for it. Before running, predict whether the held-out loss will "
        "change more or less than it did for the birds, and why.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/kandinsky-generation-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/kandinsky-generation-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weights notes: https://github.com/kurtvalcorza/kandinsky-generation-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Hugging Face decoder repository: https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder (revision `{MODEL_REVISION}`)\n"
        "- Hugging Face prior repository: https://huggingface.co/kandinsky-community/kandinsky-2-2-prior (revision 9fc51ad5732afc5d031724219d22e6c42179c5a8)\n"
        "- Shakhmatov, A., et al. (2023). Kandinsky 2.2: https://github.com/ai-forever/Kandinsky-2\n"
        "- Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. ICLR: https://arxiv.org/abs/2106.09685\n"
        "- Cherti, M., et al. (2023). Reproducible scaling laws for contrastive language-image learning. CVPR (the LAION CLIP scorer): https://arxiv.org/abs/2212.07143\n"
        "- DIMER Notebook Specification 2.2 and Model Card Specification 1.2 (in the ml-worker repository)\n"
    ),
}
