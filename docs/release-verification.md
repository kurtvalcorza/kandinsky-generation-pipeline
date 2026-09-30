# Release verification

`tutorials/kandinsky_generation_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.2 (REL8). This file is the durable release-gate record.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version,
  the standalone carrier and the isolated environment; `metadata.dimer` declares that profile, spec `2.2`, a §3.3
  pedagogical mode, `standalone: true`, `requires_dimer_worker: false` and `generated_from` (repository, revision,
  package-module SHA-256, per-file hashes, generator);
- the carrier (ST1–ST8, SRC4, PAR1–PAR4): exactly one carrier cell whose `CARRIED_FILES` equal the repository files
  named by the template (the four package modules, `tools/tutorial_stages.py`, `tutorials/requirements-colab.lock.txt`,
  the three snapshot manifests, `LICENSE`) plus the generated `source.json`; every `CARRIED_HASHES` entry is the
  SHA-256 of its text and is recorded in the cell and notebook metadata; the cell writes each file and raises on a
  hash mismatch; the notebook is byte-identical (on LF) to `tools/build_notebook.py` output for its recorded revision;
- the lock (ENV1, ENV2): it pins every `pyproject.toml` runtime pin at the same version, every entry carries
  `--hash=sha256:`, and its header records `--generate-hashes --only-binary :all:` and the manylinux x86_64 target;
- the isolated install (RUN10, ENV6, §25.13): no kernel cell runs `pip` except `uv pip install --python <isolated
  env> --require-hashes`; no `sys.executable`, `importlib`, `pip install` or `-m pip`; kernel imports limited to the
  standard library, `IPython.display` and `google.colab`; downloads (`urllib`) only in the install cell; `UV_URL`,
  `UV_BYTES` and a 64-hex `UV_SHA256` equal to the template's pinned `uv` wheel; `--managed-python` CPython 3.12.12,
  `--only-binary :all:`, the Hugging Face token and `PYTHONPATH` removal, `MPLBACKEND='Agg'`, the CUDA check in the
  isolated environment, and a `run_stage` that re-raises the stage's own error type and message;
- the learner path: the stages called in order (`weights`, `prepare`, `encode`, `frozen`, `adapt`, `evaluate`,
  `reload`, then the gated `activity`); both BYOD form fields exactly as `USE_BYOD = False  # @param
  {type:"boolean"}` and `BYOD_PATH = ''  # @param {type:"string"}` in the cell that runs `prepare` (EXE1/EXE2);
  `USE_BYOD` and `RUN_ACTIVITY` each assigned once; no learner prose that asks for a runtime restart;
- the carried stage runner's required calls (`KandinskyPipeline.from_pretrained(..., use_lora=True)`, staging and
  verification of the three snapshots, `fetch_sample_dataset`, `load_byod_dataset`, `dataset_manifest`,
  `write_dataset_csv`, `validate_dataset` with the refusal probes, the dataset-digest re-check, `encode_prompts`,
  `release_prior` and the prompt cache, the frozen evaluation, generation and CLIP scoring with
  `real_photo_baseline`, `pipe.adapt` with its explicit hyperparameters, the in-memory reference values,
  `save_artifact`, the fresh-process `from_artifact` evaluation with the guaranteed checks, the second fresh-process
  reload with the parity check, the new-prompt generation, the provenance fields `safetensors_only: True`,
  `remote_code_executed: False` and the data base URL, the error record), and the expected exports;
- `MODEL_ID`/`MODEL_REVISION` never rebound in a kernel cell, the revision absent from every kernel cell (it lives in
  the carried package and manifests), a 40-hex immutable commit, and the same identity string in `README.md`,
  `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions;
- forbidden patterns in the kernel and in every carried file: credential-in-URL, `git clone` / `github.com`, an
  editable install, a mutable `revision='main'`, `trust_remote_code=True`, unsafe deserialization, `extractall`,
  magics, `--no-binary` / `--no-build-isolation` / `--trusted-host` / `--extra-index-url`; and in the kernel only, a
  repository-package import;
- the guided layer (GDL1–GDL15), the four infrastructure cells titled and collapsed, the learner cells not collapsed,
  and the optional activity gated by `RUN_ACTIVITY = False` after the reload stage;
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.2"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_samples.py`, `tests/test_adaptation.py`, `tests/test_role_helpers.py`,
`tests/test_import_boundary.py`, `tests/test_notebook_parity.py`, `tests/test_tutorial_stages.py`). In CI, which has no
`torch`, `test_tutorial_stages.py` runs its kernel-side tests (the generated carrier writes and verifies every file, and
`run_stage` re-raises an invalid BYOD zip's refusal text) and skips the CPU pre-flight.

## CPU pre-flight of the stage runner (not runtime evidence)

On 2026-09-30, on Windows (CPython 3.12.12, `torch 2.14.0+cpu`, `diffusers 0.40.0`, `transformers 5.17.0`,
`peft 0.21.0`, in a scratch virtual environment), `tests/test_tutorial_stages.py` passed 6 of 6:

- the generated carrier cell wrote the 11 carried files and verified each against `CARRIED_HASHES`;
- the generated `run_stage`, driving the carried `tutorial_stages.py` in a subprocess, raised
  `RuntimeError: Stage 'prepare' failed (exit 2): ValueError: captions.csv is missing columns ['caption']` for a zip
  without a `caption` column, and `… ValueError: records[0]: image sides must be within 256..4096 px, got (200, 200)`
  for 200 × 200 images;
- every stage (`weights`, `prepare`, `encode`, `frozen`, `adapt`, `evaluate`, `reload`, `activity`) ran in order on
  CPU against a stub UNet/MoVQ (from `tests/test_adaptation.py`), a stub prior, a stub CLIP scorer, a stub
  generator and 12 synthetic records, each stage building its own pipeline so that everything crossed over through
  the run directory; all expected exports were written, the reload parity was `denoising_mse_diff` 0.0 and
  `mean_abs_pixel_diff` 0.0, and a stage run before `prepare` was refused with `data.json is missing`.

This proves the stage plumbing and the file hand-offs only. The real snapshot staging, the real prior, UNet, MoVQ,
diffusers generation and CLIP scorer, the `uv` bootstrap and the locked install were not exercised: they need a Linux
x86_64 GPU runtime.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab GPU runtime (T4 or better, ≥ 15 GB) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh Linux x86_64 GPU container; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | WSL workstation GPU (RTX 5070 Ti laptop, 12 GB usable VRAM), sequential cell executor with a `google.colab` shim, pre-staged pins and snapshots | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new Linux x86_64 GPU runtime (Colab, or a fresh-container executor above)
   with **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshots `weights/kandinsky-2-2-decoder/`, `weights/kandinsky-2-2-prior/`, `weights/clip-vit-b-32-laion2b/`
   or the data cache `weights/inat-birds/`; the runtime needs about 30 GB of free disk and a GPU of at least 15 GB;
3. run the notebook top-to-bottom with `Run all` and **no runtime restart**, without editing implementation cells
   (form parameters at their defaults: `USE_BYOD = False`, `BYOD_PATH = ''`, `STEPS = 20`, `GUIDANCE_SCALE = 4.0`,
   `IMAGES_PER_PROMPT = 2`, `EPOCHS = 4`, `LEARNING_RATE = 1e-4`, `BATCH_SIZE = 1`, `RUN_ACTIVITY = False`);
4. verify that the Section 2 carrier reports the revision recorded in `metadata.dimer.generated_from`, and that the
   isolated environment reports CPython 3.12.12 and the locked versions (`torch 2.14.0`, `diffusers 0.40.0`,
   `transformers 5.17.0`, `peft 0.21.0`) with `'cuda': True`, and that the kernel's own packages were not changed;
5. verify every default-path stage completes;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime, outcome, and observed metrics in the tables below;
8. record no access tokens or other secrets.

## Manual clean-runtime evidence

No run of the isolated-environment notebook (generator `build_notebook.py/3.0`) is recorded yet. Every run below is of
the previous notebook, which pip-installed its pins into the kernel; it is evidence for the stage logic it shared, not
for this revision's environment bootstrap.

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `kandinsky_generation_colab.ipynb` (`E2E`) | `b674640` / blob `34711f6de24f` | 2026-09-29 | Kaggle T4 clean container, fresh interpreter, empty Hugging Face cache, no repository checkout | **PASS** — default `Run all` path and the REL12 BYOD journey (positive and two negative inputs) at the same revision; see *Recorded executions* |
| `kandinsky_generation_colab.ipynb` (`E2E`), earlier revision | `6347e25` / blob `dd19501b4aa2` | 2026-09-29 | Kaggle T4 clean container, fresh interpreter, empty Hugging Face cache, no repository checkout | **PASS** — default `Run all` path, 12/12 code cells; see *Recorded executions* |
| `kandinsky_generation_colab.ipynb` | local pre-flight | 2026-09-24 | Local pre-flight harness (WSL, CPython 3.12.3, CUDA RTX 5070 Ti laptop, `google.colab` shim) | PASS — local feasibility and unit test suite verified; pre-flight only, **not** promotion evidence |

## Recorded executions

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-30 | `256fcb219551e07ac56ac084cf8410a9126c04b4` / blob `26a6d01839ffaded22e2316b05b1573f5cc0b434` | Google Colab, Tesla T4 (15,360 MiB); kernel Python 3.13.15, isolated environment: Python 3.12.12, `torch 2.14.0+cu130`, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0`, 68 hash-locked packages; environment set up in 98 s | Default `Run all` from a fresh runtime, form fields at their defaults | — | **PASSED** — 11/11 code cells in one pass (execution counts 1–11), 0 errors, no restart. Held-out test `denoising_mse` 0.077038 (frozen) → 0.07613 (adapted); `label_accuracy` 0.8333 → 0.8333 (real-photo ceiling 0.9167); `reference_similarity` 68.910 → 69.204 (ceiling 88.660). Fresh-process reload parity: `denoising_mse_diff` 0.0, `mean_abs_pixel_diff` 0.0. The executed notebook's cell sources are identical to the committed notebook. Sample-sanity evidence, not a benchmark |
| 2026-09-29 | `256fcb219551e07ac56ac084cf8410a9126c04b4` / blob `26a6d01839ffaded22e2316b05b1573f5cc0b434` | Kaggle batch kernel, Tesla T4 (15,360 MiB), isolated environment: Python 3.12.12, `torch 2.14.0+cu130`, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0`, 68 hash-locked packages; notebook fetched at the commit and blob-verified, run with `nbclient` in **strict single-pass mode** (a restart request fails the run), Hugging Face cache empty at start | Default `Run all` path | 795.8 s | **PASSED** — 11/11 code cells in one pass, 0 errors. Held-out test `denoising_mse` 0.077038 → 0.076139; `label_accuracy` 0.8333 → 0.8333; reload parity 0.0 / 0.0 |
| 2026-09-29 | `256fcb219551e07ac56ac084cf8410a9126c04b4` / blob `26a6d01839ffaded22e2316b05b1573f5cc0b434` | as above, strict single pass | REL12 BYOD journey: in the executed copy only, `USE_BYOD = True` and `BYOD_PATH` = a zip built in the kernel from 12 pinned CC0 iNaturalist photographs (6 Northern Cardinal, 6 Blue Jay); then the committed BYOD cell was re-run against two incompatible zips | 630.7 s | **PASSED** — 12/12 code cells in one pass, 0 errors. Positive: 12 records split 8 / 2 / 2 and carried through every stage to a fresh-process reload (parity 0.0 / 0.0). Negative: `Stage 'prepare' failed (exit 2): ValueError: captions.csv is missing columns ['caption']` and `Stage 'prepare' failed (exit 2): ValueError: records[0]: image sides must be within 256..4096 px, got (200, 200)`, raised in the kernel before any model ran on them |
| 2026-09-29 | `b6746407aad2566fc32cb13ecd947165c29781c0` / blob `34711f6de24fa858d913c5d89f94e100e1294cee` | Kaggle batch kernel, Tesla T4 (15,360 MiB), Python 3.12.13, `torch 2.14.0+cu130`, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0`; notebook fetched at the commit and blob-verified, run in a fresh interpreter with `nbclient`, Hugging Face cache empty at start | Default `Run all` path, form fields at their defaults (`USE_BYOD = False`, `BYOD_PATH = ''`, `RUN_ACTIVITY = False`) | 906.8 s | **PASSED** — 12/12 code cells, 0 errors; the install cell's restart guard fired once (preloaded `numpy`, `protobuf`, `cuda-bindings`) and the kernel was restarted and re-run from the top. Held-out test `denoising_mse` 0.077038 (frozen) → 0.076142 (adapted); `label_accuracy` 0.8333 → 0.8333 (real-photo ceiling 0.9167); `reference_similarity` 68.910 → 69.087 (ceiling 88.660). Reload parity: `denoising_mse_diff` 0.0, `mean_abs_pixel_diff` 0.0. One run on 12 held-out photographs; sample-sanity evidence, not a benchmark |
| 2026-09-29 | `b6746407aad2566fc32cb13ecd947165c29781c0` / blob `34711f6de24fa858d913c5d89f94e100e1294cee` | Kaggle batch kernel, Tesla T4 (15,360 MiB), Python 3.12.13, `torch 2.14.0+cu130`, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0`; notebook fetched at the commit and blob-verified, run in a fresh interpreter with `nbclient`, Hugging Face cache empty at start | REL12 BYOD journey. In the executed copy only (not committed), the Section 4 form was set to `USE_BYOD = True` and `BYOD_PATH` = a zip built in the kernel from 12 CC0 research-grade iNaturalist photographs (6 Northern Cardinal, 6 Blue Jay, 12 observers, not in the sample corpus), each checked against a pinned SHA-256; the committed cell source was checked by SHA-256 before the edit. After `Run all`, one appended harness cell re-ran the committed Section 4 source against two incompatible zips | 688.7 s | **PASSED** — 13/13 code cells, 0 errors. Positive: 12 records split 8 / 2 / 2 by caption and ran through validation, prompt encoding, frozen evaluation, LoRA fine-tuning, adapted evaluation, generation, export and reload (`denoising_mse_diff` 0.0, `mean_abs_pixel_diff` 0.0); held-out test `denoising_mse` 0.104984 → 0.104758. Negative: a `captions.csv` without `caption` was refused with `captions.csv is missing columns ['caption']`, and a 200 × 200 image was refused with `records[0]: image sides must be within 256..4096 px, got (200, 200)`, both before any model ran on them. With one test photograph per caption, the real-photo `reference_similarity` is 100.0 by construction; the BYOD numbers show that the path runs, not that it performs |
| 2026-09-29 | `6347e25db6477378de0f9d536596ef2de2df4c70` / blob `dd19501b4aa2793608f916622cd2b3f4aae2887b` | Kaggle batch kernel, Tesla T4 (15,360 MiB), Python 3.12.13, `torch 2.14.0+cu130`, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0`; notebook fetched at the commit and blob-verified, run in a fresh interpreter with `nbclient`, Hugging Face cache empty at start | Default `Run all` path, form fields at their defaults (`USE_BYOD = False`, `RUN_ACTIVITY = False`): stage and verify three snapshots (16,473 MB) → sample → validate → encode prompts and release prior → frozen evaluation → LoRA fine-tuning → adapted evaluation → new prompt → export → fresh reload | 848.6 s | **PASSED** — 12/12 code cells, 0 errors. The install cell's restart guard fired once because the Kaggle kernel had preloaded `numpy 2.0.2`, `protobuf 5.29.5` and `cuda-bindings 12.9.4`; the executor restarted the kernel and re-ran from the top, as a user would. Held-out test `denoising_mse` 0.077038 (frozen) → 0.076140 (adapted); `label_accuracy` 0.8333 → 0.8333 (real-photo ceiling 0.9167); `reference_similarity` 68.910 → 69.397 (ceiling 88.660); `clip_prompt_similarity` 33.218 → 32.235 (real photos 30.250). Reload parity: `denoising_mse_diff` 0.0, `mean_abs_pixel_diff` 0.0. Adapter `adapter.safetensors` 6,607,664 bytes, SHA-256 `6a49cc95…`. One run on 12 held-out photographs; sample-sanity evidence, not a benchmark |
| 2026-09-24 | local feasibility | Local GPU harness (WSL, CPython 3.12.3, CUDA RTX 5070 Ti laptop 12 GB) | Stage → load decoder + prior → generate 512² image (20 steps, guidance 4.0) | 14.70 s | **PASSED** — peak VRAM allocated: 3,224.4 MB (res: 3,904.0 MB); CPU offload needed: False; pre-flight feasibility evidence |

## Current status

**Release-grade.** At `256fcb2` (notebook blob `26a6d01839ff`) the notebook runs every stage in an isolated hash-locked environment and installs nothing into the kernel. Its default `Run all` path passed in one pass on Google Colab (T4) and on a clean Kaggle T4 in strict single-pass mode, and its BYOD branch passed the REL12 journey (representative photographs accepted and carried through adaptation, evaluation, export and a fresh-process reload; two incompatible inputs refused with the validator's message). The earlier rows record the previous notebook, which needed a manual restart on hosted runtimes.
