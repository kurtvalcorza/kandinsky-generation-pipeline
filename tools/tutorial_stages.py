"""Stage runner for the standalone Kandinsky 2.2 tutorial (NOTEBOOK_SPEC 2.2 §25.13 isolated-environment pattern).

The tutorial notebook carries this file verbatim (as ``tutorial_stages.py`` in its run directory, beside the carried
package under ``src/``) and runs every stage with the interpreter of an isolated, hash-locked environment::

    python -u tutorial_stages.py --root RUN_DIR --weights WEIGHTS_DIR --stage prepare [--byod PATH]

Nothing is installed into the notebook kernel. Each stage is a separate process, so a stage starts from files only:
the verified snapshots under ``--weights``, the dataset recorded by ``prepare``, the prompt-embedding cache written by
``encode``, the adapter artifact written by ``adapt`` and the JSON records of earlier stages. Learner-facing exports go
to ``RUN_DIR/outputs``; hand-off state goes to ``RUN_DIR/state``. On failure a stage writes
``RUN_DIR/state/<stage>.error.json`` with the exception type and message, which the notebook re-raises in the kernel.

Stages: weights → prepare → encode → frozen → adapt → evaluate → reload, plus the optional ``activity``.
"""
# ruff: noqa: E501  -- the printed dictionaries are the learner-facing output; they are kept on one line each
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

STEM = "kandinsky_generation"
EVAL_SEED = 0
GENERATION_SEED = 1000
NEW_PROMPT_SEED = 2000
PARITY_SEED = 3000
NEW_PROMPT = "a photo of a House Finch (Haemorhous mexicanus) perched on a snow-covered branch in winter"
SNAPSHOT_KEYS = ("kandinsky-2-2-decoder", "kandinsky-2-2-prior", "clip-vit-b-32-laion2b")
SAMPLE_CACHE = "inat-birds"
PROMPT_CACHE = "prompt_cache.safetensors"
PROMPT_INDEX = "prompt_cache.json"


# --------------------------------------------------------------------------------------------------
# run context and small helpers
# --------------------------------------------------------------------------------------------------


class Run:
    """Paths of one run: carried sources and state under ``root``, snapshots under ``weights``."""

    def __init__(self, root: Path, weights: Path, options: argparse.Namespace) -> None:
        self.root = root
        self.weights = weights
        self.options = options
        self.out = root / "outputs"
        self.state = root / "state"
        self.out.mkdir(parents=True, exist_ok=True)
        self.state.mkdir(parents=True, exist_ok=True)

    def snapshot(self, key: str) -> Path:
        return self.weights / key

    def write_state(self, name: str, value: Any) -> Path:
        path = self.state / name
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        return path

    def read_state(self, name: str, needed_by: str) -> Any:
        path = self.state / name
        if not path.is_file():
            raise RuntimeError(f"{name} is missing: run the stage that writes it before '{needed_by}' (run the notebook from the top)")
        return json.loads(path.read_text(encoding="utf-8"))

    def write_output(self, name: str, value: Any) -> Path:
        path = self.out / name
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        return path


def gpu_memory_gb() -> float | None:
    import torch

    return round(torch.cuda.memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None


def grid(images: list[Any], path: Path, columns: int = 6) -> Path:
    from PIL import Image

    tiles = [im.resize((256, 256)) for im in images]
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (256 * columns, 256 * rows), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (256 * (i % columns), 256 * (i // columns)))
    sheet.save(path)
    return path


def without_images(generation: dict[str, Any]) -> dict[str, Any]:
    """A generation result without its PIL images (JSON-serialisable)."""
    return {**generation, "images": [{k: v for k, v in g.items() if k != "image"} for g in generation["images"]]}


def runtime_versions() -> dict[str, Any]:
    import diffusers
    import peft
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "cuda": torch.cuda.is_available(),
    }


# --------------------------------------------------------------------------------------------------
# model and data factories (the CPU pre-flight test replaces these with stubs)
# --------------------------------------------------------------------------------------------------


def load_pipeline(run: Run) -> Any:
    import torch

    from kandinsky_generation_pipeline import KandinskyPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return KandinskyPipeline.from_pretrained(
        weights_dir=run.snapshot(SNAPSHOT_KEYS[0]), prior_dir=run.snapshot(SNAPSHOT_KEYS[1]), device=device, use_lora=True
    )


def load_from_artifact(run: Run, artifact_dir: Path) -> Any:
    import torch

    from kandinsky_generation_pipeline import KandinskyPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return KandinskyPipeline.from_artifact(
        artifact_dir, weights_dir=run.snapshot(SNAPSHOT_KEYS[0]), prior_dir=run.snapshot(SNAPSHOT_KEYS[1]), device=device
    )


def load_scorer(run: Run, device: str) -> Any:
    from kandinsky_generation_pipeline import ClipScorer

    return ClipScorer(weights_dir=run.snapshot(SNAPSHOT_KEYS[2]), device=device)


def load_sample_splits(run: Run) -> dict[str, list[dict[str, Any]]]:
    from kandinsky_generation_pipeline import fetch_sample_dataset

    return fetch_sample_dataset(cache_dir=run.weights / SAMPLE_CACHE)


def stage_snapshots(run: Run) -> list[dict[str, Any]]:
    """Install the carried manifests, fetch the absent files at the pinned revisions and verify every file."""
    from kandinsky_generation_pipeline import (
        MANIFEST_NAME,
        MODEL_ID,
        MODEL_LICENSE,
        MODEL_REVISION,
        PRIOR_ID,
        PRIOR_LICENSE,
        PRIOR_REVISION,
        SCORER_ID,
        SCORER_LICENSE,
        SCORER_REVISION,
        stage_missing_files,
        stage_missing_prior_files,
        stage_missing_scorer_files,
        verify_prior_snapshot,
        verify_scorer_snapshot,
        verify_snapshot,
    )

    plan = (
        (SNAPSHOT_KEYS[0], MODEL_ID, MODEL_REVISION, MODEL_LICENSE, stage_missing_files, verify_snapshot),
        (SNAPSHOT_KEYS[1], PRIOR_ID, PRIOR_REVISION, PRIOR_LICENSE, stage_missing_prior_files, verify_prior_snapshot),
        (SNAPSHOT_KEYS[2], SCORER_ID, SCORER_REVISION, SCORER_LICENSE, stage_missing_scorer_files, verify_scorer_snapshot),
    )
    records = []
    for key, model_id, revision, license_name, stage, verify in plan:
        carried = run.root / "weights" / key / MANIFEST_NAME
        manifest = json.loads(carried.read_text(encoding="utf-8"))
        if (manifest["modelId"], manifest["revision"]) != (model_id, revision):
            raise RuntimeError(f"carried {key} manifest does not name the identity pinned by the package; regenerate the notebook")
        target = run.snapshot(key)
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(carried, target / MANIFEST_NAME)
        print({"model_id": model_id, "revision": revision, "license": license_name, "files": len(manifest["files"]), "total_bytes": manifest["totalBytes"]}, flush=True)
        fetched = stage(target, allow_download=True)
        print({"weights_dir": str(target), "fetched": fetched}, flush=True)
        verified = verify(target)
        print({"verified_files": len(verified["files"]), "revision": verified["revision"]}, flush=True)
        records.append({"key": key, "id": model_id, "revision": revision, "license": license_name, "files": len(verified["files"]), "fetched": fetched})
    return records


def load_data(run: Run, stage: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Rebuild the splits recorded by `prepare` from their source and refuse if they changed."""
    from kandinsky_generation_pipeline import dataset_manifest, load_byod_dataset, split_dataset

    data = run.read_state("data.json", stage)
    splits = split_dataset(load_byod_dataset(data["byod_path"]), seed=0) if data["source"] == "byod" else load_sample_splits(run)
    splits = {name: splits[name] for name in ("train", "validation", "test")}
    digest = dataset_manifest(splits)["digest"]
    if digest != data["dataset_digest"]:
        raise RuntimeError(f"the dataset changed since 'prepare' (digest {digest[:16]}… != {data['dataset_digest'][:16]}…); re-run from Section 4")
    return splits, data


def save_prompt_cache(run: Run, cache: dict[str, Any]) -> None:
    import safetensors.torch

    prompts = list(cache)
    tensors = {}
    for i, prompt in enumerate(prompts):
        tensors[f"{i}.image_embeds"] = cache[prompt]["image_embeds"].contiguous()
        tensors[f"{i}.negative_image_embeds"] = cache[prompt]["negative_image_embeds"].contiguous()
    safetensors.torch.save_file(tensors, run.state / PROMPT_CACHE)
    run.write_state(PROMPT_INDEX, {"prompts": prompts})


def load_prompt_cache(run: Run, pipe: Any, stage: str) -> int:
    import safetensors.torch

    prompts = run.read_state(PROMPT_INDEX, stage)["prompts"]
    tensors = safetensors.torch.load_file(run.state / PROMPT_CACHE)
    cache = {p: {"image_embeds": tensors[f"{i}.image_embeds"], "negative_image_embeds": tensors[f"{i}.negative_image_embeds"]} for i, p in enumerate(prompts)}
    return pipe.import_prompt_cache(cache)


# --------------------------------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------------------------------


def stage_weights(run: Run) -> None:
    """Section 3: install the carried manifests, fetch the absent files at the pinned revisions, verify every file."""
    snapshots = stage_snapshots(run)
    run.write_output("weights.json", {"snapshots": snapshots})


def stage_prepare(run: Run) -> None:
    """Section 4: fetch the sample (or read the BYOD zip), validate, split, and probe the refusals."""
    from PIL import Image

    from kandinsky_generation_pipeline import (
        INPUT_SCHEMA,
        SAMPLE_LABEL_SOURCE,
        dataset_manifest,
        load_byod_dataset,
        sample_prompts,
        split_dataset,
        validate_dataset,
        validate_inputs,
        write_dataset_csv,
    )

    byod = run.options.byod
    if byod:
        byod_path = Path(byod).resolve()
        splits = split_dataset(load_byod_dataset(byod_path), seed=0)
        data_source = "BYOD (" + byod_path.name + ")"
    else:
        byod_path = None
        splits = load_sample_splits(run)
        data_source = SAMPLE_LABEL_SOURCE
    train_records, val_records, test_records = splits["train"], splits["validation"], splits["test"]

    dataset_report = dataset_manifest({"train": train_records, "validation": val_records, "test": test_records})
    print({"data_source": data_source, "splits": {k: v["n_records"] for k, v in dataset_report["splits"].items()}, "captions": dataset_report["splits"]["train"]["n_captions"], "disjoint": dataset_report["disjoint"]})
    print({"shorter_side": dataset_report["splits"]["train"]["shorter_side"], "centre_cropped": dataset_report["splits"]["train"]["centre_cropped"], "digest": dataset_report["digest"][:16] + "..."})
    print({"first_test_record": validate_inputs(test_records[0]), "caption": test_records[0]["caption"]})
    prompts = sample_prompts(train_records)
    print({"prompts": prompts})
    sample_csv = write_dataset_csv(test_records, run.out / f"{STEM}_sample_captions.csv")
    print({"sample_csv": str(sample_csv)})

    print({"validation": INPUT_SCHEMA["validation"]})
    probes = {
        "missing caption": [{"id": r["id"], "image": r["image"]} for r in train_records[:4]],
        "image too small": [{**train_records[0], "image": Image.new("RGB", (200, 200))}, *train_records[1:4]],
        "duplicate id": [train_records[0], *train_records[:4]],
    }
    probe_results = []
    for name, records in probes.items():
        try:
            validate_dataset(records)
            probe_results.append({"probe": name, "verdict": "accepted"})
        except (TypeError, ValueError) as exc:
            probe_results.append({"probe": name, "rejected": str(exc)[:110]})
        print(probe_results[-1])

    run.write_state(
        "data.json",
        {
            "source": "byod" if byod_path else "sample",
            "byod_path": str(byod_path) if byod_path else None,
            "data_source": data_source,
            "dataset_digest": dataset_report["digest"],
            "ids": {name: [r["id"] for r in records] for name, records in splits.items()},
        },
    )
    run.write_output("prepare.json", {"data_source": data_source, "dataset": dataset_report, "prompts": prompts, "probes": probe_results})


def stage_encode(run: Run) -> None:
    """Section 5: encode every prompt with the prior, release it, and write the prompt-embedding cache."""
    from kandinsky_generation_pipeline import sample_prompts

    splits, _data = load_data(run, "encode")
    pipe = load_pipeline(run)
    all_prompts = sample_prompts(splits["train"] + splits["validation"] + splits["test"]) + [NEW_PROMPT]
    encode_report = pipe.encode_prompts(all_prompts)
    print({**encode_report, "gpu_memory_gb_with_prior": gpu_memory_gb()})
    released = pipe.release_prior()
    print({"prior_released": released, "gpu_memory_gb_after_release": gpu_memory_gb(), "device": pipe.device, "precision": str(pipe.dtype).replace("torch.", "")})
    cache = pipe.export_prompt_cache()
    save_prompt_cache(run, cache)
    print({"prompt_cache": str(run.state / PROMPT_CACHE), "prompts": len(cache)})
    run.write_output("encode.json", {**encode_report, "prior_released": released, "cached_prompts": list(cache), "device": pipe.device})


def stage_frozen(run: Run) -> None:
    """Section 6: the frozen model's held-out denoising loss and CLIP-scored generations, and the real-photo ceiling."""
    from kandinsky_generation_pipeline import real_photo_baseline, sample_prompts, score_generations

    opts = run.options
    splits, data = load_data(run, "frozen")
    val_records, test_records = splits["validation"], splits["test"]
    prompts = sample_prompts(splits["train"])
    pipe = load_pipeline(run)
    load_prompt_cache(run, pipe, "frozen")
    scorer = load_scorer(run, pipe.device)
    t0 = time.perf_counter()
    frozen_val = pipe.evaluate(val_records, seed=EVAL_SEED)
    frozen_test = pipe.evaluate(test_records, seed=EVAL_SEED)
    print({"frozen_denoising_mse": {"validation": frozen_val["denoising_mse"], "test": frozen_test["denoising_mse"]}, "by_timestep_test": frozen_test["by_timestep"], "seconds": round(time.perf_counter() - t0, 1)})

    generation_prompts = [p for p in prompts for _ in range(opts.images_per_prompt)]
    frozen_generation = pipe.generate(generation_prompts, seed=GENERATION_SEED, steps=opts.steps, guidance_scale=opts.guidance)
    print({"generated": len(frozen_generation["images"]), "steps": frozen_generation["steps"], "guidance_scale": frozen_generation["guidance_scale"], "seconds": frozen_generation["seconds"], "adapted": frozen_generation["model"]["adapted"]})
    frozen_scores = score_generations(scorer, frozen_generation["images"], references=test_records)
    real_ceiling = real_photo_baseline(scorer, test_records)
    print({"frozen_generations": {k: frozen_scores[k] for k in ("clip_prompt_similarity", "label_accuracy", "reference_similarity")}})
    print({"real_photo_ceiling": {k: real_ceiling[k] for k in ("clip_prompt_similarity", "label_accuracy", "reference_similarity")}})
    for entry in frozen_scores["per_image"][:: opts.images_per_prompt]:
        print({"prompt": entry["prompt"][:42], "clip": entry["clip_prompt_similarity"], "nearest": entry["nearest_prompt"][13:40], "correct": entry["correct"], "reference_similarity": entry["reference_similarity"]})
    grid_path = grid([g["image"] for g in frozen_generation["images"]], run.out / f"{STEM}_frozen_grid.jpg")
    print({"grid": str(grid_path)})
    record = {
        "data_source": data["data_source"],
        "generation": {"prompts": generation_prompts, "steps": opts.steps, "guidance_scale": opts.guidance, "images_per_prompt": opts.images_per_prompt, "seed": GENERATION_SEED},
        "validation": frozen_val,
        "test": frozen_test,
        "generation_result": without_images(frozen_generation),
        "generations": frozen_scores,
        "real_photo_ceiling": real_ceiling,
    }
    run.write_state("frozen.json", record)
    run.write_output("frozen.json", record)


def stage_adapt(run: Run) -> None:
    """Section 7: bounded LoRA fine-tuning; record the in-memory model's reference values; export the adapter."""
    from kandinsky_generation_pipeline import sample_prompts

    opts = run.options
    splits, data = load_data(run, "adapt")
    frozen = run.read_state("frozen.json", "adapt")
    settings = frozen["generation"]
    pipe = load_pipeline(run)
    load_prompt_cache(run, pipe, "adapt")

    def report(entry: dict[str, Any]) -> None:
        row = {"epoch": entry["epoch"], "train_loss": None if entry["train_loss"] is None else round(entry["train_loss"], 4), "val_denoising_mse": entry["val_loss"]}
        if "note" in entry:
            row["note"] = entry["note"]
        print(row, flush=True)

    t0 = time.perf_counter()
    adapt_result = pipe.adapt(splits["train"], splits["validation"], epochs=opts.epochs, lr=opts.lr, batch_size=opts.batch_size, seed=EVAL_SEED, progress=report)
    adapt_seconds = round(time.perf_counter() - t0, 1)
    print({"trainable_parameters": adapt_result["adapter"]["n_trainable"], "total_parameters": adapt_result["adapter"]["n_total"], "steps": adapt_result["steps"], "best_epoch": adapt_result["best_epoch"], "precision": adapt_result["adapter"]["precision"], "seconds": adapt_seconds, "gpu_memory_gb": gpu_memory_gb()})

    # Reference values of the trained, in-memory model, for the fresh-process reload check in Section 9 (VER2/VER4).
    in_memory_test = pipe.evaluate(splits["test"], seed=EVAL_SEED)
    parity_prompt = sample_prompts(splits["train"])[0]
    parity_image = pipe.generate([parity_prompt], seed=PARITY_SEED, steps=settings["steps"], guidance_scale=settings["guidance_scale"])["images"][0]["image"]
    parity_image.save(run.state / "parity_in_memory.png")

    artifact_dir = run.out / f"{STEM}_adapter"
    shutil.rmtree(artifact_dir, ignore_errors=True)
    manifest = pipe.save_artifact(artifact_dir, metadata={"tutorial": STEM, "data_source": data["data_source"]})
    print({"artifact": str(artifact_dir), "format": manifest["format"], "tensors": manifest["weights"]["n_tensors"], "bytes": manifest["weights"]["bytes"], "sha256": manifest["weights"]["sha256"][:16] + "..."})
    record = {
        "history": adapt_result["history"],
        "best_epoch": adapt_result["best_epoch"],
        "adaptation": {k: v for k, v in adapt_result.items() if k not in ("history", "trainable_names")},
        "adaptation_seconds": adapt_seconds,
        "hyperparameters": {"epochs": opts.epochs, "lr": opts.lr, "batch_size": opts.batch_size, "seed": EVAL_SEED},
        "in_memory": {"test_denoising_mse": in_memory_test["denoising_mse"], "parity_prompt": parity_prompt, "parity_seed": PARITY_SEED, "parity_image": "state/parity_in_memory.png"},
        "artifact": {"dir": str(artifact_dir), "sha256": manifest["weights"]["sha256"], "bytes": manifest["weights"]["bytes"], "tensors": manifest["weights"]["n_tensors"]},
    }
    run.write_state("adapt.json", record)
    run.write_output("adapt.json", record)


def stage_evaluate(run: Run) -> None:
    """Section 8: a fresh process loads the exported adapter and scores it exactly as the frozen model was scored."""
    from kandinsky_generation_pipeline import MODEL_ID, MODEL_KEY, MODEL_REVISION, PRIOR_ID, PRIOR_REVISION, score_generations

    splits, data = load_data(run, "evaluate")
    val_records, test_records = splits["validation"], splits["test"]
    frozen = run.read_state("frozen.json", "evaluate")
    adapted_state = run.read_state("adapt.json", "evaluate")
    settings = frozen["generation"]
    pipe = load_from_artifact(run, Path(adapted_state["artifact"]["dir"]))
    load_prompt_cache(run, pipe, "evaluate")
    scorer = load_scorer(run, pipe.device)
    adapted_val = pipe.evaluate(val_records, seed=EVAL_SEED)
    adapted_test = pipe.evaluate(test_records, seed=EVAL_SEED)
    adapted_generation = pipe.generate(settings["prompts"], seed=settings["seed"], steps=settings["steps"], guidance_scale=settings["guidance_scale"])
    adapted_scores = score_generations(scorer, adapted_generation["images"], references=test_records)
    frozen_val, frozen_test, frozen_scores, real_ceiling = frozen["validation"], frozen["test"], frozen["generations"], frozen["real_photo_ceiling"]
    comparison = {
        "denoising_mse_validation": {"frozen": frozen_val["denoising_mse"], "adapted": adapted_val["denoising_mse"]},
        "denoising_mse_test": {"frozen": frozen_test["denoising_mse"], "adapted": adapted_test["denoising_mse"]},
        "denoising_mse_test_by_timestep": {t: {"frozen": frozen_test["by_timestep"][t], "adapted": adapted_test["by_timestep"][t]} for t in adapted_test["by_timestep"]},
        "clip_prompt_similarity": {"frozen": frozen_scores["clip_prompt_similarity"], "adapted": adapted_scores["clip_prompt_similarity"], "real_photos": real_ceiling["clip_prompt_similarity"]},
        "label_accuracy": {"frozen": frozen_scores["label_accuracy"], "adapted": adapted_scores["label_accuracy"], "real_photos": real_ceiling["label_accuracy"]},
        "reference_similarity": {"frozen": frozen_scores["reference_similarity"], "adapted": adapted_scores["reference_similarity"], "real_photos": real_ceiling["reference_similarity"]},
    }
    for name, row in comparison.items():
        print({name: row})
    step = settings["images_per_prompt"]
    for before, after in zip(frozen_scores["per_image"][::step], adapted_scores["per_image"][::step], strict=True):
        print({"prompt": before["prompt"][:42], "reference_similarity": {"frozen": before["reference_similarity"], "adapted": after["reference_similarity"]}, "correct": {"frozen": before["correct"], "adapted": after["correct"]}})
    grid_path = grid([g["image"] for g in adapted_generation["images"]], run.out / f"{STEM}_adapted_grid.jpg")
    print({"grid": str(grid_path)})
    evaluation_report = {
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": MODEL_KEY},
        "components": {"id": PRIOR_ID, "revision": PRIOR_REVISION},
        "scorer": frozen_scores["scorer"],
        "data_source": data["data_source"],
        "dataset": json.loads((run.out / "prepare.json").read_text(encoding="utf-8"))["dataset"],
        "generation": {k: settings[k] for k in ("steps", "guidance_scale", "images_per_prompt", "seed")},
        "frozen": {"validation": frozen_val, "test": frozen_test, "generations": frozen_scores},
        "adapted": {"validation": adapted_val, "test": adapted_test, "generations": adapted_scores, "loaded_from": "exported artifact, fresh process"},
        "real_photo_ceiling": real_ceiling,
        "comparison": comparison,
        "adaptation": adapted_state["adaptation"],
        "history": adapted_state["history"],
        "adaptation_seconds": adapted_state["adaptation_seconds"],
    }
    report_path = run.write_output(f"{STEM}_evaluation_report.json", evaluation_report)
    history = adapted_state["history"]
    best = history[adapted_state["best_epoch"]]
    # Guaranteed by the procedure: epoch 0 (the frozen model) is a candidate, so the kept validation loss is no worse.
    if not best["val_loss"] <= history[0]["val_loss"]:
        raise AssertionError(f"kept epoch validation loss {best['val_loss']} exceeds the frozen model's {history[0]['val_loss']}")
    # The exported adapter, loaded in this fresh process, reproduces the kept epoch's validation loss.
    if not abs(adapted_val["denoising_mse"] - best["val_loss"]) < 1e-4:
        raise AssertionError(f"the exported adapter gives validation loss {adapted_val['denoising_mse']}, the kept epoch recorded {best['val_loss']}")
    print({"test_denoising_mse_change": round(adapted_test["denoising_mse"] - frozen_test["denoising_mse"], 6), "note": "held-out observation, not asserted"})
    print({"report": str(report_path)})
    run.write_state("evaluate.json", {"comparison": comparison, "adapted_test_denoising_mse": adapted_test["denoising_mse"], "adapted_generations": adapted_scores, "generation_seconds": adapted_generation["seconds"]})


def stage_reload(run: Run) -> None:
    """Section 9: another fresh process rebuilds the pipeline from the artifact, checks parity with the trained
    in-memory model recorded by `adapt`, renders the new prompt, and writes the result record."""
    import numpy as np
    from PIL import Image

    from kandinsky_generation_pipeline import (
        CORPUS_BASE_URL,
        CORPUS_LICENSE,
        MODEL_LICENSE,
        PRIOR_LICENSE,
        SCORER_LICENSE,
        score_generations,
    )

    splits, data = load_data(run, "reload")
    frozen = run.read_state("frozen.json", "reload")
    adapted_state = run.read_state("adapt.json", "reload")
    evaluated = run.read_state("evaluate.json", "reload")
    settings = frozen["generation"]
    artifact_dir = Path(adapted_state["artifact"]["dir"])
    artifact_manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    print({"artifact": str(artifact_dir), "format": artifact_manifest["format"], "tensors": artifact_manifest["weights"]["n_tensors"], "bytes": artifact_manifest["weights"]["bytes"], "sha256": artifact_manifest["weights"]["sha256"][:16] + "..."})

    reloaded = load_from_artifact(run, artifact_dir)
    load_prompt_cache(run, reloaded, "reload")
    reloaded_test = reloaded.evaluate(splits["test"], seed=EVAL_SEED)
    in_memory = adapted_state["in_memory"]
    after = reloaded.generate([in_memory["parity_prompt"]], seed=in_memory["parity_seed"], steps=settings["steps"], guidance_scale=settings["guidance_scale"])["images"][0]["image"]
    with Image.open(run.root / in_memory["parity_image"]) as saved:
        before = np.asarray(saved.convert("RGB"), dtype=np.float32)
    parity = {
        "denoising_mse_diff": round(abs(reloaded_test["denoising_mse"] - in_memory["test_denoising_mse"]), 8),
        "mean_abs_pixel_diff": round(float(np.abs(before - np.asarray(after, dtype=np.float32)).mean()), 4),
        "compared_with": "the trained in-memory model of the 'adapt' process",
    }
    print({"reload_parity": parity, "reloaded_best_epoch": reloaded.adapter["best_epoch"]})
    if not (parity["denoising_mse_diff"] < 1e-6 and parity["mean_abs_pixel_diff"] < 1.0):
        raise AssertionError(f"reload parity failed: {parity}")

    scorer = load_scorer(run, reloaded.device)
    new_generation = reloaded.generate([NEW_PROMPT, NEW_PROMPT], seed=NEW_PROMPT_SEED, steps=settings["steps"], guidance_scale=settings["guidance_scale"])
    new_scores = score_generations(scorer, new_generation["images"])
    print({"new_prompt": NEW_PROMPT, "clip_prompt_similarity": new_scores["clip_prompt_similarity"], "seconds": new_generation["seconds"], "note": "sanity check, not an evaluation"})
    for i, entry in enumerate(new_generation["images"]):
        entry["image"].save(run.out / f"{STEM}_new_prompt_{i}.png")

    source_path = run.root / "source.json"
    notebook_source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.is_file() else None
    weights = json.loads((run.out / "weights.json").read_text(encoding="utf-8"))
    report = json.loads((run.out / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8"))
    result_payload = {
        "notebook_source": notebook_source,
        "repository_revision": notebook_source["revision"] if notebook_source else None,
        "model": {**report["model"], "model_license": MODEL_LICENSE, "device": reloaded.device, "precision": str(reloaded.dtype).replace("torch.", ""), "source": reloaded.source},
        "components": {**report["components"], "license": PRIOR_LICENSE},
        "scorer": {**report["scorer"], "license": SCORER_LICENSE},
        "provenance": {
            "snapshots": {s["key"]: s["files"] for s in weights["snapshots"]},
            "safetensors_only": True,
            "remote_code_executed": False,
            "prior_released_before_training": json.loads((run.out / "encode.json").read_text(encoding="utf-8"))["prior_released"],
            "data_base_url": CORPUS_BASE_URL,
            "data_license": CORPUS_LICENSE,
        },
        "runtime": {**runtime_versions(), "environment": "isolated hash-locked environment (one process per stage)"},
        "data_source": data["data_source"],
        "comparison": evaluated["comparison"],
        "artifact": {"dir": str(artifact_dir), "sha256": artifact_manifest["weights"]["sha256"], "bytes": artifact_manifest["weights"]["bytes"]},
        "reload_parity": parity,
        "new_prompt": {"prompt": NEW_PROMPT, "clip_prompt_similarity": new_scores["clip_prompt_similarity"]},
    }
    run.write_output(f"{STEM}_result.json", result_payload)
    run.write_output("reload.json", {"reload_parity": parity, "new_prompt": result_payload["new_prompt"]})
    print("outputs/:")
    for path in sorted(run.out.rglob("*")):
        if path.is_file():
            print(f"  - {path.relative_to(run.root).as_posix()} ({path.stat().st_size / 1024:.1f} KB)")


def stage_activity(run: Run) -> None:
    """Section 10 (optional): regenerate the Section 8 prompt/seed pairs at another guidance scale."""
    from kandinsky_generation_pipeline import score_generations

    splits, _data = load_data(run, "activity")
    frozen = run.read_state("frozen.json", "activity")
    adapted_state = run.read_state("adapt.json", "activity")
    evaluated = run.read_state("evaluate.json", "activity")
    settings = frozen["generation"]
    guidance = run.options.guidance
    pipe = load_from_artifact(run, Path(adapted_state["artifact"]["dir"]))
    load_prompt_cache(run, pipe, "activity")
    scorer = load_scorer(run, pipe.device)
    activity_generation = pipe.generate(settings["prompts"], seed=settings["seed"], steps=settings["steps"], guidance_scale=guidance)
    activity_scores = score_generations(scorer, activity_generation["images"], references=splits["test"])
    real_ceiling = frozen["real_photo_ceiling"]
    adapted_scores = evaluated["adapted_generations"]
    rows = {}
    for key in ("clip_prompt_similarity", "label_accuracy", "reference_similarity"):
        rows[key] = {f"guidance {settings['guidance_scale']}": adapted_scores[key], f"guidance {guidance}": activity_scores[key], "real_photos": real_ceiling[key]}
        print({key: rows[key]})
    seconds = {f"guidance {settings['guidance_scale']}": evaluated["generation_seconds"], f"guidance {guidance}": activity_generation["seconds"]}
    print({"seconds": seconds})
    grid_path = grid([g["image"] for g in activity_generation["images"]], run.out / f"{STEM}_activity_grid.jpg")
    print({"grid": str(grid_path)})
    run.write_output("activity.json", {"guidance_scale": guidance, "comparison": rows, "seconds": seconds, "generations": activity_scores})


STAGES = {
    "weights": stage_weights,
    "prepare": stage_prepare,
    "encode": stage_encode,
    "frozen": stage_frozen,
    "adapt": stage_adapt,
    "evaluate": stage_evaluate,
    "reload": stage_reload,
    "activity": stage_activity,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="run directory holding the carried sources")
    parser.add_argument("--weights", type=Path, required=True, help="directory holding the three snapshots and the photo cache")
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--byod", default="", help="prepare: a BYOD zip or directory instead of the sample")
    parser.add_argument("--steps", type=int, default=20, help="frozen: generation steps")
    parser.add_argument("--guidance", type=float, default=4.0, help="frozen/activity: classifier-free guidance scale")
    parser.add_argument("--images-per-prompt", type=int, default=2, help="frozen: images per caption")
    parser.add_argument("--epochs", type=int, default=4, help="adapt: training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="adapt: AdamW learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="adapt: records per optimiser step")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    root = options.root.resolve()
    carried_src = root / "src"
    if carried_src.is_dir() and str(carried_src) not in sys.path:
        sys.path.insert(0, str(carried_src))
    run = Run(root, options.weights.resolve(), options)
    error_file = run.state / f"{options.stage}.error.json"
    error_file.unlink(missing_ok=True)
    started = time.perf_counter()
    try:
        STAGES[options.stage](run)
    except Exception as exc:  # the notebook re-raises this message in the kernel
        traceback.print_exc()
        message = str(exc) or repr(exc)
        error_file.write_text(json.dumps({"stage": options.stage, "type": type(exc).__name__, "message": message}), encoding="utf-8")
        print(f"STAGE FAILED ({options.stage}): {type(exc).__name__}: {message}", flush=True)
        return 2
    print({"stage": options.stage, "status": "ok", "seconds": round(time.perf_counter() - started, 1)}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
