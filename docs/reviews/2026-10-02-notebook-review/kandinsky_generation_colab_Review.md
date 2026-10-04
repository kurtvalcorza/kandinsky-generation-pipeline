# Kandinsky 2.2 text-to-image guided notebook — Review

**Verdict: Needs revision**  
**Review date:** 3 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/kandinsky-generation-pipeline`  
**Notebook:** `tutorials/kandinsky_generation_colab.ipynb`  
**Reviewed commit:** `b75e501337ebdc05c2c16e5c295ec0acdff44250` (`main`)  
**Notebook Git blob:** `26a6d01839ffaded22e2316b05b1573f5cc0b434`  
**Finding prefix:** `KGN`  
**Framework:** Notebook Review Framework v1; requirements baseline DIMER Notebook Specification **2.2** (2026-09-26, `ml-worker` `origin/main`)

## Executive assessment

This is a well-built notebook. Every stage runs in an isolated hash-locked `uv` environment, so nothing is installed
into the kernel. The code is carried byte for byte with hash checks, and three snapshots are pinned by commit and
digest. The notebook scores a generator without calling any number "quality": it reports a paired held-out denoising
loss on identical noise, CLIP scores, a validation-selected epoch that may be epoch 0, and a fresh-process reload with
parity. The default `Run all` path and the REL12 BYOD journey both have hosted evidence **at this exact notebook blob**,
and both passed in one pass with no restart.

The sibling notebook's per-timestep error does not recur here. Row 33 (`kandinsky-controlnet-depth`, KCD-M1) said the
denoising loss rises with the timestep. This notebook says it is "usually highest at the smallest timesteps", and the
hosted run agrees: 0.284 at t = 100 falling to 0.0007 at t = 900 (probe `P05`).

The Major finding (KGN-M1) concerns the reference the notebook calls the **real-photo ceiling**. It is the reference
for learning objective 3, the question in Section 8 and the conclusion template, but it is a ceiling on only one of its
three measures. On CLIP prompt similarity the generated images score **above** it (33.2 against 30.3). On reference
similarity the real photographs are scored against a mean that **includes each photograph itself**, which inflates the
value to 88.7. A leave-one-out estimate from the same run is about 57.7, below the generations' 68.9. In BYOD with one
test photograph per caption, that value is 100.0 by construction. The notebook discloses none of this, and the metric's
docstring says the opposite of what the code does.

Four Minor findings follow, three of which leave an applicable spec **MUST** unmet: SPL3 (the split's independence
assumption is unstated, and "hold out by caption" names the wrong split), DAT9 (pretraining overlap of the public
sample is not stated) and DAT12 (the BYOD minimum the notebook states is refused by the code). The fourth covers three
"not yet recorded" statements that the release record contradicts.

None of these findings says the default run fails. The hosted evidence shows it passes.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Revision | `b75e501` (merge of PR #3). The notebook blob equals the blob of the recorded hosted runs at `256fcb2`; `256fcb2..b75e501` changes only `MODEL_CARD.md`, `README.md`, `STATUS.md`, `docs/release-verification.md` and `tutorials/README.md` (probe `P02`). The carrier records generating revision `fc00107` |
| Profile / mode | `E2E` / `GUIDED`, declared in the opening and `metadata.dimer` |
| Spec declared / applied | 2.2 / 2.2 |
| Audience | Can open a hosted notebook, run cells and read short Python; no prior experience with diffusion models assumed |
| Prerequisites stated | Linux x86_64 CUDA GPU ≥ 15 GB (T4), about 30 GB disk; CPU-only explicitly unsupported |
| Supported runtime | Google Colab T4 (primary), Kaggle T4, Linux Jupyter with CUDA |
| Promised outcomes | Three digest-verified snapshots; 60 CC0 bird photographs validated and split 36 / 12 / 12 with refusal probes; prompts encoded and the prior released; frozen baseline (held-out denoising loss, CLIP prompt similarity, label accuracy, reference similarity) beside a "real-photo ceiling"; bounded LoRA (rank 8, 1,646,592 parameters); paired comparison from the exported adapter; fresh-process reload with parity; a new prompt; an optional guidance-scale activity; BYOD through the same stages |
| Generator | `tools/build_notebook.py` (3.0) from `tools/notebook_template.py`; stage logic in `tools/tutorial_stages.py` (carried) |

### Evidence actually obtained

**Source inspection.** All 40 cells (11 code, 29 markdown), the carried stage runner, `samples.split_dataset`,
`metrics.score_generations` / `real_photo_baseline`, the template, `README.md`, `tutorials/README.md`,
`docs/release-verification.md` and `STATUS.md`.

**Documented execution evidence (this exact blob).** `docs/release-verification.md` records three runs of blob
`26a6d018…`: Google Colab T4 (2026-09-30, 11/11 code cells, no restart), a strict single-pass Kaggle T4 run
(2026-09-29, 795.8 s) and the REL12 BYOD journey (positive zip plus two refusals, 630.7 s). The Kaggle executor outputs
are archived under `.agent/backups/kandinsky-kaggle-2026-09-29/out-pilot/`. The `run_summary.json` files there name
commit `256fcb2` and blob `26a6d018…`, with `runner.ok: true` and one pass each. This review read both runs'
`kandinsky_generation_evaluation_report.json` (sha256 in `source_manifest.json`) for the values used below. The executed
Colab notebook itself was not re-read.

**Direct execution (CPU, Windows, CPython 3.12.14, numpy 2.5.3, Pillow 12.3.0; no torch, no diffusers, no model
weights).** 12 probes in `run_probes.py` covering: notebook parse, compile and hygiene; blob identity; the isolated-install
pattern; carrier parity against the repository (10 carried files compared, 0 mismatches); the repository's offline
suite (33 passed, 3 skipped for lack of `diffusers`; `PYTHONPATH=src` because the package is not installed in the probe
interpreter); `tools/validate_release_assets.py` (PASS); `tools/build_notebook.py --check` (up to date); and the carried
BYOD loader and splitter on synthetic zips. These static and validator-only runs are **not** execution evidence under
REL8. Probe wall-clock time was well under the 20-minute cap.

**Not verified.** Any GPU stage in this review; the Section 10 guidance activity on real weights (the release record's
CPU stub pre-flight is the only evidence); learner understanding.

### Journeys

| Journey | Evidence basis | Result |
|---|---|---|
| First-time learner | Source inspection | Well scaffolded: audience, how-to-use, I/O table, roadmap with a fast path, glossary, predictions, What to notice, Check your reasoning, troubleshooting, a conclusion template. The loss-by-timestep explanation is correct. The learner is misled by the "real-photo ceiling" (KGN-M1) |
| Clean default | Documented execution evidence at this blob | Passed on Colab T4 and strict Kaggle T4, one pass, no restart, reload parity 0.0 / 0.0. Not re-run here |
| Active learning | Source inspection; CPU stub pre-flight in the release record | The activity is cleanly isolated: it writes only `outputs/activity.json` and nothing reads it back (probe `P11`), so reruns cannot leave a stale block in another export (row 33's KCD-m5 does not recur). Not verified on real weights |
| Reuse and recovery | Documented REL12 evidence at this blob; direct CPU execution of the loader and splitter | Positive BYOD and two refusals documented. The stated minimum dataset size is refused by the code (KGN-m3). On BYOD the reference-similarity "ceiling" is 100.0 by construction (KGN-M1). Reload from files with parity is documented |

## 2. Separate judgments

| Dimension | Judgment |
|---|---|
| Technical correctness | Sound. The environment is isolated, the carrier matches the repository, snapshots are digest-pinned, only safetensors are loaded, each stage runs in its own process, evaluation is paired and seeded, the epoch is chosen on validation, and the fresh-process reload has explicit parity. One defect: `real_photo_baseline` includes each photograph in its own reference mean, contrary to its docstring (KGN-M1) |
| Promise fulfilment | Every promised stage runs and is evidenced at this blob, including BYOD. The "ceiling" comparison is promised (objective 3, Section 8, conclusion), but the reference used does not support it as described (KGN-M1) |
| Scientific validity | Comparisons are paired and labelled sample-sanity, and the guaranteed result is kept apart from observations. Gaps: the real-photo reference (KGN-M1), the split's independence assumption (KGN-m1) and pretraining overlap (KGN-m2) |
| Learner experience | Above fleet average. The timestep explanation is correct. The ceiling framing teaches a wrong reading of two of the three CLIP measures (KGN-M1) |
| Spec conformance | RUN, ST, ENV, MOD, VER, REL and GDL requirements are met on the evidence above. **Unmet MUSTs:** EVAL3 (KGN-M1: a principal reference is not explained as what it measures), SPL3 (KGN-m1), DAT9 (KGN-m2) and DAT12 (KGN-m3, the stated limit is inaccurate) |

## 3. Findings

### KGN-M1 — Major (spec MUST EVAL3; EVAL10): the "real-photo ceiling" is exceeded on one measure and inflated by self-matching on another

**Location:** opening (cell 0, "the same CLIP scores on the real photographs themselves — the ceiling"); learning
objective 3 ("**compare** the frozen model with the real-photo ceiling"); glossary (cell 3, "roughly the best a generator
could reach on these measures"); Section 6 (cell 23, "`real_photo_baseline` scores the real test photographs the same
way: the ceiling" and the *Predict*: "which CLIP measure … sit furthest below the real-photo ceiling"); Section 6 answer
(cell 25, "a generated image can score close to the ceiling"); Section 8 question (cell 29, "do its generations move
towards the real-photo ceiling?"); conclusion template (cell 39, "against a real-photo ceiling of [ceiling values]").
Generator: `tools/notebook_template.py` lines 117, 244, 434, 445, 525, 534. Code:
`src/kandinsky_generation_pipeline/metrics.py` `real_photo_baseline` (line 131).

**Observed issue:** the reference is a ceiling on only one of its three measures.

| Measure | Frozen | Adapted | "Ceiling" (real photos) | What the "ceiling" actually is |
|---|---|---|---|---|
| CLIP prompt similarity (default run) | 33.218 | 32.334 | **30.25** | Below the generations. A generator renders the prompt more literally than a field photograph does |
| CLIP prompt similarity (BYOD run) | 35.47 | 35.161 | **26.485** | Below the generations |
| Label accuracy (default) | 0.8333 | 0.8333 | 0.9167 | A meaningful reference |
| Reference similarity (default) | 68.91 | 69.469 | **88.66** | Each real photo is scored against the normalised mean of its caption's 2 test photos **including itself**. Inverting r = √((1 + c) / 2) for unit embeddings gives a leave-one-out photo-to-photo similarity of about **57.7**, below the generations |
| Reference similarity (BYOD) | 54.861 | 55.495 | **100.0** | One test photo per caption: each photo is compared with itself, so the value is 100 by construction |

The code's own note says "references include each photo itself", but the docstring says "references = the other
records" and the notebook says neither (probe `P06`). The leave-one-out figure compares a photograph with one other
photograph while a generation is compared with the mean of two, so it is only an indicative bound. The self-match
inflation itself is arithmetic: with two references per caption, the photograph contributes half of its own reference
mean.

**Consequence:** this misleads on the comparison that objective 3, the Section 6 prediction, the Section 8 question and
the conclusion template are built around. A learner reading the default run will conclude that the generations sit far
below real photographs on reference similarity (68.9 against 88.7), which is largely an artefact. The same learner will
find prompt similarity "above the ceiling" with no explanation. In Section 8 the adapted prompt similarity falls from
33.2 to 32.3, which looks like movement "towards the real-photo ceiling" but is a loss of prompt alignment. A BYOD user
will write "against a ceiling of 100.0" into the conclusion. The glossary's definition ("roughly the best a generator
could reach") is false for two of the three rows.

**Evidence:** documented execution evidence (default and BYOD `evaluation_report.json`, blob `26a6d018…`), source
inspection of `metrics.py`, and arithmetic on the recorded per-image values (probe `P06`).

**Recommended correction:** score the real photographs leave-one-out: each photo against the mean of the **other**
test photos of its caption, and omit captions with a single test photo, reporting the count. Fix the docstring to
match. Rename the reference to **real-photo reference** throughout the template, and say per measure what it means.
Label accuracy is a meaningful reference. Generated images can exceed it on prompt similarity, because a generator
renders the prompt more literally than a field photograph. On reference similarity it is "how similar real photos of a
species are to each other". Rewrite the Section 6 prediction, the Section 8 question and the conclusion template so that
none of them asks whether the generations reach a ceiling.

**Acceptance check:** no learner-facing text calls the real-photo values a ceiling or "the best a generator could
reach". `real_photo_baseline` never includes a photo in its own reference mean (a unit test with two references per
caption gives the pairwise cosine), and a BYOD run with one test photo per caption does not report a reference
similarity of 100.0. `tools/build_notebook.py --check` passes.

### KGN-m1 — Minor (spec MUST SPL3; SPL10): the random split's independence assumption is unstated, and "hold out by caption" names the opposite split

**Location:** Section 4 (cell 17) and its answer (cell 19); opening BYOD paragraph (cell 0, "Your records are split by
caption"); *Interpretation and limits* (cell 39): "**Hold out by caption, not by image:** the split keeps every
caption's images across sets". Generator: `tools/notebook_template.py` lines 71 and 714;
`samples.split_dataset` docstring ("grouped by caption"). `tutorials/README.md` *Split integrity* note.

**Observed issue:** the sample and the BYOD split are both seeded shuffles **stratified within caption**, so every
caption appears in train, validation and test (probe `P07`: 3 captions, all 3 in every split). "Hold out by caption"
names a group split that holds whole captions out, which is the opposite mechanism; only the clause after it is right.
The word "independent" appears nowhere in the notebook, so SPL3 is unmet. The sample corpus mitigates the risk by
design (cell 19: "one photograph per observer per species"), but nothing tells a BYOD user to do the same, and BYOD
de-duplicates only byte-identical images. A one-pixel variant of a photo was kept beside its original (probe `P07`).
`tutorials/README.md` claims "the notebook states that … photographs of the same individual or observer should be
grouped in real data", and the notebook contains no such statement.

**Consequence:** a learner applying the transfer advice may build the wrong split, and a BYOD user is not warned that
burst shots or crops of one scene inflate the held-out numbers.

**Evidence:** source inspection plus direct CPU execution of `split_dataset` on synthetic images.

**Recommended correction:** in Section 4 and the BYOD text, state that a random split assumes the photographs are
independent, and that near-duplicates (bursts, crops, the same individual or scene) should be removed or kept on one
side. Rename the transfer point to "**Stratify by caption**: every caption has held-out images, so the test measures
new photographs of seen captions, not unseen captions". Fix the `split_dataset` docstring and the `tutorials/README.md`
claim.

**Acceptance check:** the notebook states the independence assumption before the split runs. No text says "hold out by
caption" for a within-caption stratified split. The BYOD contract mentions near-duplicates. `tutorials/README.md` claims
nothing the notebook does not say.

### KGN-m2 — Minor (spec MUST DAT9): possible pretraining overlap of the public sample is not stated

**Location:** Prerequisites (cell 4), Section 4 (cell 17), *Interpretation and limits* (cell 39).

**Observed issue:** the default sample is 60 public iNaturalist photographs. The CLIP scorer
(`laion/CLIP-ViT-B-32-laion2B-s34B-b79K`) was trained on LAION-2B web images, and Kandinsky 2.2 on web-scale image–text
data. Overlap with these public photographs cannot be ruled out, and no markdown cell says so (probe `P08`).

**Consequence:** the label accuracy, the reference similarity and the real-photo values may be inflated by memorisation,
and the learner is not told. DAT9 is an applicable MUST.

**Evidence:** source inspection.

**Recommended correction:** add one sentence to *Interpretation and limits* and the Section 4 data note. The sample
photographs are public web images that the generator's and the scorer's training data may include, so the CLIP numbers
on them may be optimistic; BYOD on private photographs avoids this.

**Acceptance check:** a learner-facing cell states the pretraining-overlap limitation for the default sample.

### KGN-m3 — Minor (spec MUST DAT12): the stated BYOD minimum is accepted by the text but refused by the code

**Location:** opening *Bring Your Own Data* (cell 0): "at least four images, and at least one caption with three or
more images". Generator: `tools/notebook_template.py` line 70. Code: `samples.split_dataset` with
`MIN_TRAIN_RECORDS = 4` and 20 % / 20 % per-caption validation and test shares. The troubleshooting row (cell 38: "at
least four training images in total") and the transfer prompt ("at least three images per caption") each describe a
different rule.

**Observed issue:** zips that satisfy the stated contract are refused (probe `P10`, direct execution of the carried
loader and splitter on synthetic 320 px PNGs):

| Layout | Meets stated contract | Result |
|---|---|---|
| 4 images, 1 caption | yes | refused: `split leaves 2 training records; at least 4 are required` |
| 5 images, 1 caption | yes | refused: `split leaves 3 training records; …` |
| 4 images, captions 3 + 1 | yes | refused: `split leaves 2 training records; …` |
| 6 images, 2 captions × 3 | yes | refused: `split leaves 2 training records; …` |
| 6 images, 1 caption | yes | accepted (4 / 1 / 1) |
| 8 images, 2 captions × 4 | yes | accepted (4 / 2 / 2) |

**Consequence:** a user who prepares the documented minimum gets a refusal. The message names the rule and the
troubleshooting row explains it, so this is friction rather than a dead end, but DAT12 requires the limits to be stated
correctly before upload.

**Evidence:** direct execution (CPU, loader and splitter only; no model) and source inspection.

**Recommended correction:** state the rule the code enforces in one place, and use the same wording in the opening, the
troubleshooting row and the transfer prompt. For example: "at least 4 training images after a per-caption split that
keeps about 20 % for validation and 20 % for test, so at least 6 images in practice; only captions with three or more
images contribute held-out images".

**Acceptance check:** every layout that meets the stated minimum is accepted by `split_dataset(load_byod_dataset(...))`,
and the three places state the same rule.

### KGN-m4 — Minor (UX12): three "not yet recorded" statements contradict the release record

**Location:** opening *Run all* paragraph (cell 0): "its duration has not yet been recorded" (generator
`tools/notebook_template.py` line 65). `tutorials/README.md` registry row, *Run-all* column: "not yet recorded for this
revision". `docs/release-verification.md` *Manual clean-runtime evidence* (line 113): "No run of the isolated-environment
notebook … is recorded yet", with a table that lists only the earlier revisions.

**Observed issue:** the same file's *Recorded executions* table records 795.8 s for this blob on a strict Kaggle T4, a
Colab T4 pass and the BYOD journey, and its *Current status* says Release-grade (probe `P09`).

**Consequence:** a learner planning a session is told no measurement exists. A release reviewer reading the
manual-evidence section finds it contradicting the status. Both are localised.

**Evidence:** source inspection.

**Recommended correction:** state "about 13 minutes on a Kaggle T4 (795.8 s, measured 2026-09-29) plus downloads; your
runtime will vary" in the template and the registry row. Add the `256fcb2` runs to the manual-evidence table and remove
the "recorded yet" sentence.

**Acceptance check:** no document says the duration or the run is unrecorded while the release record holds a
measurement for the same blob.

### KGN-S1 — Suggestion: state the chance level beside label accuracy

Label accuracy is an argmax over the distinct captions: chance is 1/6 ≈ 0.167 on the sample and 1/2 on the two-caption
BYOD run, where every model scored 1.0. One printed `chance_level` field and one sentence in Section 6 would give the
learner the trivial baseline that EVAL10/EVAL11 recommend.

### KGN-S2 — Suggestion: point Section 8 at the per-timestep paired change

The notebook correctly explains that the loss is highest at the smallest timesteps, but the comparison prose reads only
the mean, which is about 74 % the t = 100 term (probe `P05`). One sentence pointing at
`denoising_mse_test_by_timestep` would show the learner where adaptation moved the loss: in the recorded run about
−0.0015 at t = 100, −0.0018 at t = 300 and −0.00003 at t = 900.

## 4. Readiness

**Needs revision.** There is no Blocker, and the default and BYOD journeys have hosted evidence at this exact blob with
no restart. One Major remains (KGN-M1), and four applicable MUSTs are unmet: EVAL3 (KGN-M1), SPL3 (KGN-m1), DAT9 (KGN-m2)
and DAT12 (KGN-m3). KGN-M1 needs a small code change to `real_photo_baseline` plus template prose. The rest are prose or
docstring changes. Because the notebook bytes would change, a regenerated notebook needs a new hosted `Run all` record
before it returns to Release-grade.

## 5. Verified versus inferred

- **Verified:** the notebook blob equals the hosted-evidence blob; carrier parity; no kernel install; the static gates
  pass; the per-timestep losses, CLIP values and per-image reference similarities come from the executor's own reports
  for this blob; the BYOD minimum-size boundary and the near-duplicate behaviour were directly executed on CPU; the
  self-inclusion in `real_photo_baseline` is read from the code and its own note.
- **Inferred:** the leave-one-out estimate (57.7) is derived arithmetically from the recorded values, assuming unit-norm
  embeddings, which the code applies. It was not recomputed from embeddings. The guidance activity's behaviour on real
  weights is not verified.
- **Only Kurt can confirm:** whether the intended BYOD contract is the code's 20 % / 20 % per-caption split (fix the
  text) or the text (fix the code).
- **Most likely to be wrong:** KGN-M1's severity. A reader may accept "ceiling" as loose shorthand and rate it Minor,
  as row 33 did for the prompt-similarity half (KCD-m1). The self-match inflation of reference similarity, and the
  100.0 on BYOD, are what move it to Major here.
