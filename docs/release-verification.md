# Release verification

`tutorials/kandinsky_generation_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.2 (REL8). This file is the durable release-gate record.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.2`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `samples.py`, `metrics.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST`, `PRIOR_MANIFEST` and `SCORER_MANIFEST` equal to the three
  committed snapshot manifests and the inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook
  byte-identical (on LF) to `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with
  its restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before staging), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files` / `stage_missing_prior_files` /
  `stage_missing_scorer_files` with `allow_download=True`, the three `verify_*_snapshot` calls,
  `KandinskyPipeline.from_pretrained(weights_dir=..., prior_dir=..., use_lora=True)`, `fetch_sample_dataset` from
  the pinned cache path, `load_byod_dataset`, `dataset_manifest`, `write_dataset_csv`, `validate_dataset` with the
  refusal probes, `pipe.encode_prompts` and `pipe.release_prior`, `pipe.evaluate` and `pipe.generate` +
  `score_generations` + `real_photo_baseline` on the frozen model, `pipe.adapt` with its explicit hyperparameters,
  `pipe.evaluate` after adaptation with the guaranteed assertions (kept-epoch validation loss ≤ frozen; re-scored validation loss matches the history), the new-prompt generation, `pipe.save_artifact`,
  `KandinskyPipeline.from_artifact` + `import_prompt_cache` and the reload-parity assertion, and the provenance
  fields `safetensors_only: True`, `remote_code_executed: False` and the data base URL), the six expected `outputs/`
  paths, the learner-facing statements (three pinned snapshots, prior pipeline can be released,
  generation has no ground truth, denoising loss, real-photo ceiling, not a human judgement, Apache-2.0,
  sample-sanity, CC0) and the gated-off BYOD default; forbidden patterns (credential-in-URL, any `git clone` /
  `github.com` / repository import on the primary path, a mutable `revision='main'`, direct `huggingface_hub` /
  `safetensors` / `urllib` / `diffusers` / `transformers` / `peft` / `CLIPModel` use,
  `torch.load(` / `pickle.load` / `Unpickler`, `torch.no_grad(` / `torch.inference_mode(` / `.backward(` outside
  the carried module cells, `trust_remote_code=True`, `add_adapter(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.2"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_samples.py`, `tests/test_adaptation.py`, `tests/test_role_helpers.py`,
`tests/test_import_boundary.py`, `tests/test_notebook_parity.py`).

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab GPU runtime (T4 or better, ≥ 15 GB) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | WSL workstation GPU (RTX 5070 Ti laptop, 12 GB usable VRAM), sequential cell executor with a `google.colab` shim, pre-staged pins and snapshots | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new GPU runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshots `weights/kandinsky-2-2-decoder/`, `weights/kandinsky-2-2-prior/`, `weights/clip-vit-b-32-laion2b/`
   or the data cache `weights/inat-birds/`; the runtime needs about 25 GB of free disk and a GPU of at least 15 GB;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `STEPS = 20`, `GUIDANCE_SCALE = 4.0`, `IMAGES_PER_PROMPT = 2`, `EPOCHS = 4`,
   `LEARNING_RATE = 1e-4`, `BATCH_SIZE = 1`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `diffusers==0.40.0`, `transformers==5.17.0`,
   `peft==0.21.0`, `accelerate==1.15.0`, `safetensors==0.8.0`, `huggingface-hub==1.32.0`, `numpy==2.5.3`, `pillow==11.3.0`;
5. verify every default-path stage completes;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime, outcome, and observed metrics in the tables below;
8. record no access tokens or other secrets.

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `kandinsky_generation_colab.ipynb` (`E2E`) | `b674640` / blob `34711f6de24f` | 2026-09-29 | Kaggle T4 clean container, fresh interpreter, empty Hugging Face cache, no repository checkout | **PASS** — default `Run all` path and the REL12 BYOD journey (positive and two negative inputs) at the same revision; see *Recorded executions* |
| `kandinsky_generation_colab.ipynb` (`E2E`), earlier revision | `6347e25` / blob `dd19501b4aa2` | 2026-09-29 | Kaggle T4 clean container, fresh interpreter, empty Hugging Face cache, no repository checkout | **PASS** — default `Run all` path, 12/12 code cells; see *Recorded executions* |
| `kandinsky_generation_colab.ipynb` | local pre-flight | 2026-09-24 | Local pre-flight harness (WSL, CPython 3.12.3, CUDA RTX 5070 Ti laptop, `google.colab` shim) | PASS — local feasibility and unit test suite verified; pre-flight only, **not** promotion evidence |

## Recorded executions

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-29 | `b6746407aad2566fc32cb13ecd947165c29781c0` / blob `34711f6de24fa858d913c5d89f94e100e1294cee` | Kaggle batch kernel, Tesla T4 (15,360 MiB), Python 3.12.13, `torch 2.14.0+cu130`, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0`; notebook fetched at the commit and blob-verified, run in a fresh interpreter with `nbclient`, Hugging Face cache empty at start | Default `Run all` path, form fields at their defaults (`USE_BYOD = False`, `BYOD_PATH = ''`, `RUN_ACTIVITY = False`) | 906.8 s | **PASSED** — 12/12 code cells, 0 errors; the install cell's restart guard fired once (preloaded `numpy`, `protobuf`, `cuda-bindings`) and the kernel was restarted and re-run from the top. Held-out test `denoising_mse` 0.077038 (frozen) → 0.076142 (adapted); `label_accuracy` 0.8333 → 0.8333 (real-photo ceiling 0.9167); `reference_similarity` 68.910 → 69.087 (ceiling 88.660). Reload parity: `denoising_mse_diff` 0.0, `mean_abs_pixel_diff` 0.0. One run on 12 held-out photographs; sample-sanity evidence, not a benchmark |
| 2026-09-29 | `b6746407aad2566fc32cb13ecd947165c29781c0` / blob `34711f6de24fa858d913c5d89f94e100e1294cee` | Kaggle batch kernel, Tesla T4 (15,360 MiB), Python 3.12.13, `torch 2.14.0+cu130`, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0`; notebook fetched at the commit and blob-verified, run in a fresh interpreter with `nbclient`, Hugging Face cache empty at start | REL12 BYOD journey. In the executed copy only (not committed), the Section 4 form was set to `USE_BYOD = True` and `BYOD_PATH` = a zip built in the kernel from 12 CC0 research-grade iNaturalist photographs (6 Northern Cardinal, 6 Blue Jay, 12 observers, not in the sample corpus), each checked against a pinned SHA-256; the committed cell source was checked by SHA-256 before the edit. After `Run all`, one appended harness cell re-ran the committed Section 4 source against two incompatible zips | 688.7 s | **PASSED** — 13/13 code cells, 0 errors. Positive: 12 records split 8 / 2 / 2 by caption and ran through validation, prompt encoding, frozen evaluation, LoRA fine-tuning, adapted evaluation, generation, export and reload (`denoising_mse_diff` 0.0, `mean_abs_pixel_diff` 0.0); held-out test `denoising_mse` 0.104984 → 0.104758. Negative: a `captions.csv` without `caption` was refused with `captions.csv is missing columns ['caption']`, and a 200 × 200 image was refused with `records[0]: image sides must be within 256..4096 px, got (200, 200)`, both before any model ran on them. With one test photograph per caption, the real-photo `reference_similarity` is 100.0 by construction; the BYOD numbers show that the path runs, not that it performs |
| 2026-09-29 | `6347e25db6477378de0f9d536596ef2de2df4c70` / blob `dd19501b4aa2793608f916622cd2b3f4aae2887b` | Kaggle batch kernel, Tesla T4 (15,360 MiB), Python 3.12.13, `torch 2.14.0+cu130`, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0`; notebook fetched at the commit and blob-verified, run in a fresh interpreter with `nbclient`, Hugging Face cache empty at start | Default `Run all` path, form fields at their defaults (`USE_BYOD = False`, `RUN_ACTIVITY = False`): stage and verify three snapshots (16,473 MB) → sample → validate → encode prompts and release prior → frozen evaluation → LoRA fine-tuning → adapted evaluation → new prompt → export → fresh reload | 848.6 s | **PASSED** — 12/12 code cells, 0 errors. The install cell's restart guard fired once because the Kaggle kernel had preloaded `numpy 2.0.2`, `protobuf 5.29.5` and `cuda-bindings 12.9.4`; the executor restarted the kernel and re-ran from the top, as a user would. Held-out test `denoising_mse` 0.077038 (frozen) → 0.076140 (adapted); `label_accuracy` 0.8333 → 0.8333 (real-photo ceiling 0.9167); `reference_similarity` 68.910 → 69.397 (ceiling 88.660); `clip_prompt_similarity` 33.218 → 32.235 (real photos 30.250). Reload parity: `denoising_mse_diff` 0.0, `mean_abs_pixel_diff` 0.0. Adapter `adapter.safetensors` 6,607,664 bytes, SHA-256 `6a49cc95…`. One run on 12 held-out photographs; sample-sanity evidence, not a benchmark |
| 2026-09-24 | local feasibility | Local GPU harness (WSL, CPython 3.12.3, CUDA RTX 5070 Ti laptop 12 GB) | Stage → load decoder + prior → generate 512² image (20 steps, guidance 4.0) | 14.70 s | **PASSED** — peak VRAM allocated: 3,224.4 MB (res: 3,904.0 MB); CPU offload needed: False; pre-flight feasibility evidence |

## Current status

**Candidate.** The notebook stops at its install cell on hosted runtimes that preload `numpy`, `protobuf` and `cuda-bindings` (Google Colab and Kaggle), because the pinned install replaces those loaded packages and the cell then asks for a manual runtime restart. Notebook Specification 2.2 RUN1 and RUN10 forbid a manual restart on the `Run all` path, so the tutorial is not release-ready. The Kaggle runs recorded in `docs/release-verification.md` completed only because the executor restarted the kernel automatically; they remain valid evidence for everything after the install cell. Found in a Colab `Run all` on 2026-09-29; the fix (an isolated, hash-locked environment for the tutorial stages) is in progress.
