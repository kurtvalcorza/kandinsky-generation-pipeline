"""Regression tests for the kandinsky_generation_colab review fixes (KGN-M1, KGN-m1..m4).

They run within CI's install budget (pytest, NumPy, Pillow; no torch, no model library, no weights): the carried
package functions with a NumPy-backed CLIP stand-in, the generated notebook's markdown, the stage runner's source and
the repository documents. One CPU pre-flight (torch + diffusers + safetensors, skipped in CI) drives every stage with
stub models on the smallest BYOD layout the notebook states. None of this is model or clean-runtime evidence.
"""
# ruff: noqa: E501  -- test cases quote notebook prose and refusal messages in full

from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pytest

# eo-notebook-test (Windows conda) trap: a NumPy matmul before torch's first import breaks every later torch import in
# the process (WinError 127). This module multiplies NumPy matrices, so torch is imported first when it exists.
with contextlib.suppress(ImportError):
    import torch  # noqa: F401

from conftest import synthetic_image
from kandinsky_generation_pipeline import (
    MIN_BYOD_IMAGES_PER_CAPTION,
    REAL_PHOTO_REFERENCE_KIND,
    load_byod_dataset,
    real_photo_baseline,
    real_photo_reference,
    split_dataset,
)
from test_tutorial_stages import _run, preflight  # noqa: F401  -- the CPU pre-flight fixture, reused

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "kandinsky_generation_colab.ipynb"
STATED_MINIMUM = "at least one caption with six or more distinct images"


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(f"_kgn_{name}", ROOT / "tools" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _markdown(nb: dict) -> str:
    return "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")


# --- KGN-M1: the real photographs are a leave-one-out reference line, not a ceiling --------------------------------


class _T(np.ndarray):
    """The few torch-tensor methods score_generations / real_photo_reference use, on NumPy."""

    def argmax(self, dim=None, **_):  # type: ignore[override]
        return np.asarray(self).argmax(axis=dim).view(_T)

    def mean(self, dim=None, **_):  # type: ignore[override]
        return np.asarray(self).mean(axis=dim).view(_T)

    def norm(self):
        return float(np.linalg.norm(np.asarray(self)))


class _Scorer:
    """CLIP stand-in: fixed unit vectors per image key and per caption."""

    identity = {"id": "stand-in", "revision": "0"}

    def __init__(self, images: dict[str, np.ndarray], texts: dict[str, np.ndarray]) -> None:
        self.images, self.texts = images, texts

    @staticmethod
    def _stack(rows):
        return np.stack([r / np.linalg.norm(r) for r in rows]).astype(np.float64).view(_T)

    def image_embeddings(self, images):
        return self._stack([self.images[k] for k in images])

    def text_embeddings(self, texts):
        return self._stack([self.texts[t] for t in texts])


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def test_real_photo_reference_excludes_each_photo_from_its_own_reference() -> None:
    a, b, c = _unit([1, 0.2, 0, 0]), _unit([0.6, 0.8, 0.1, 0]), _unit([0, 0, 1, 0.3])
    scorer = _Scorer({"a": a, "b": b, "c": c}, {"A": _unit([1, 0.5, 0, 0]), "C": _unit([0, 0, 1, 0])})
    records = [{"image": "a", "caption": "A"}, {"image": "b", "caption": "A"}, {"image": "c", "caption": "C"}]
    report = real_photo_reference(scorer, records)
    cos_ab = float(a @ b) * 100
    rows = report["per_image"]
    # two photos of a caption: each one's reference similarity is cos(a, b), not sqrt((1 + cos(a, b)) / 2)
    assert rows[0]["reference_similarity"] == rows[1]["reference_similarity"] == round(cos_ab, 3)
    assert rows[0]["reference_similarity"] != round(np.sqrt((1 + cos_ab / 100) / 2) * 100, 3)
    assert "reference_similarity" not in rows[2] and report["n_without_reference"] == 1
    assert report["reference_similarity"] == round(cos_ab, 3)
    assert report["reference_kind"] == REAL_PHOTO_REFERENCE_KIND == "leave-one-out real-photo reference"
    assert set(report["reading"]) == {"clip_prompt_similarity", "label_accuracy", "reference_similarity"}
    assert "not a ceiling" in report["note"] and "excludes the photo itself" in report["note"]
    assert real_photo_baseline is real_photo_reference  # the old name gives the corrected measure


def test_one_test_photo_per_caption_reports_no_reference_similarity_never_100() -> None:
    """The recorded BYOD run printed 100.0 because each photo was compared with itself; now there is no value."""
    a, b = _unit([1, 0.1, 0]), _unit([0, 1, 0.2])
    scorer = _Scorer({"a": a, "b": b}, {"A": _unit([1, 0, 0]), "B": _unit([0, 1, 0])})
    report = real_photo_reference(scorer, [{"image": "a", "caption": "A"}, {"image": "b", "caption": "B"}])
    assert "reference_similarity" not in report and report["n_without_reference"] == 2
    assert all("reference_similarity" not in row for row in report["per_image"])
    assert report["clip_prompt_similarity"] > 0 and report["label_accuracy"] == 1.0


def test_recorded_values_reproduce_the_reviews_leave_one_out_mean() -> None:
    """The 2026-09-29 Kaggle record's 12 self-inclusive per-photo values (two test photos per caption) give the 88.66
    the notebook printed; with cos = 2 s² − 1 the leave-one-out mean is the review's 57.7 (the value Section 6 quotes)."""
    recorded = [95.419, 95.779, 95.779, 82.961, 95.419, 82.961, 86.968, 85.623, 85.212, 85.623, 86.968, 85.212]
    assert round(sum(recorded) / len(recorded), 2) == 88.66
    loo = [(2 * (s / 100) ** 2 - 1) * 100 for s in recorded]
    assert round(sum(loo) / len(loo), 1) == 57.7
    a, b = _unit([1, 0.3, 0.2]), _unit([0.4, 1, 0])
    mean = (a + b) / np.linalg.norm(a + b)
    assert abs(float(a @ mean) - np.sqrt((1 + float(a @ b)) / 2)) < 1e-12


def test_no_learner_text_or_document_calls_the_real_photographs_a_ceiling(nb: dict) -> None:
    md = _markdown(nb)
    for phrase in ("real-photo ceiling", "the best a generator could reach", "the ceiling"):
        assert phrase not in md
    assert [m.start() for m in re.finditer("ceiling", md) if not md[: m.start()].endswith("not a ")] == []
    assert md.count("not a ceiling") >= 5 and "leave-one-out" in md and "`real_photo_reference`" in md
    assert "57.7" in md and "88.66" in md  # the worked answer explains the old self-inclusive value
    for name in ("README.md", "MODEL_CARD.md", "tutorials/README.md", "STATUS.md", "docs/release-verification.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert [m.start() for m in re.finditer("ceiling", text) if not text[: m.start()].endswith("not a ")] == [], name


def test_stage_runner_scores_the_leave_one_out_reference() -> None:
    runner = (ROOT / "tools" / "tutorial_stages.py").read_text(encoding="utf-8")
    assert "real_reference = real_photo_reference(scorer, test_records)" in runner
    assert "real_photo_baseline" not in runner and "real_ceiling" not in runner and "real_photo_ceiling" not in runner
    assert '"photos_without_a_reference": real_reference["n_without_reference"]' in runner
    # a BYOD caption with no test photograph has no reference similarity: never index it unguarded
    assert 'entry["reference_similarity"]' not in runner and 'before["reference_similarity"]' not in runner


# --- KGN-m1: independence assumption and the stratified (not grouped) split ----------------------------------------


def test_split_prose_states_independence_and_names_the_stratified_split(nb: dict) -> None:
    md = _markdown(nb)
    assert "Hold out by caption" not in md and "**Stratify by caption:**" in md
    cells = nb["cells"]
    prepare = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and "run_stage('prepare'" in _src(c))
    before = "\n".join(_src(c) for c in cells[:prepare] if c["cell_type"] == "markdown")
    assert "assumes the photographs are independent" in before  # stated before the split runs
    opening = _src(cells[0])
    assert "stratified within each caption" in opening and "near-duplicates" in opening
    assert "grouped by caption" not in (ROOT / "src" / "kandinsky_generation_pipeline" / "samples.py").read_text(encoding="utf-8")
    registry = (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    assert "should be grouped in real data" not in registry and "independent" in registry


# --- KGN-m2: pretraining overlap -----------------------------------------------------------------------------------


def test_pretraining_overlap_is_stated_for_the_default_sample(nb: dict) -> None:
    md = _markdown(nb)
    assert md.count("**Pretraining overlap.**") == 2
    assert "may be optimistic" in md and "BYOD on private photographs avoids this" in md


# --- KGN-m3: one stated BYOD minimum, accepted end to end ----------------------------------------------------------


def test_the_stated_minimum_is_said_the_same_way_in_every_place(nb: dict) -> None:
    md = _markdown(nb)
    assert md.count(STATED_MINIMUM) == 3  # opening BYOD paragraph, troubleshooting row, transfer prompt
    for stale in ("at least four images, and at least one caption with three or more images", "at least three images per caption", "provide at least four training images in total"):
        assert stale not in md
    assert MIN_BYOD_IMAGES_PER_CAPTION == 6


_POOL: list = []


def _image(i: int):
    while len(_POOL) <= i:
        _POOL.append(synthetic_image(width=256, height=256, seed=1000 + len(_POOL)))
    return _POOL[i]


@pytest.mark.parametrize("main", range(MIN_BYOD_IMAGES_PER_CAPTION, 21))
@pytest.mark.parametrize("extra", [(), (1,), (2,), (1, 2), (5,), (3, 4)])
def test_every_layout_meeting_the_stated_minimum_is_accepted_by_the_split(main: int, extra: tuple[int, ...]) -> None:
    counts = (main, *extra)
    records, k = [], 0
    for c, n in enumerate(counts):
        for _ in range(n):
            records.append({"id": f"r{k}", "image": _image(k), "caption": f"caption {c}"})
            k += 1
    splits = split_dataset(records, seed=0)
    assert len(splits["train"]) >= 4 and len(splits["test"]) >= 1


def _zip(path: Path, counts: tuple[int, ...]) -> Path:
    rows = io.StringIO()
    writer = csv.writer(rows)
    writer.writerow(("id", "file", "caption"))
    k = 0
    with zipfile.ZipFile(path, "w") as archive:
        for c, n in enumerate(counts):
            for _ in range(n):
                buffer = io.BytesIO()
                _image(k).save(buffer, format="PNG")
                archive.writestr(f"bird{k}.png", buffer.getvalue())
                writer.writerow((f"r{k}", f"bird{k}.png", f"a photo of bird kind {c}"))
                k += 1
        archive.writestr("captions.csv", rows.getvalue())
    return path


def test_the_minimum_zip_is_accepted_and_smaller_ones_are_refused_with_the_rule(tmp_path: Path) -> None:
    splits = split_dataset(load_byod_dataset(_zip(tmp_path / "min.zip", (6,))), seed=0)
    assert {k: len(v) for k, v in splits.items()} == {"test": 1, "validation": 1, "train": 4}
    # the layouts the review found refused do not meet the stated minimum, and the refusal now names it
    for counts in ((4,), (5,), (3, 1), (3, 3)):
        with pytest.raises(ValueError, match=r"give at least one caption 6 or more distinct images"):
            split_dataset(load_byod_dataset(_zip(tmp_path / f"{counts}.zip", counts)), seed=0)


# --- KGN-m4: the recorded duration -----------------------------------------------------------------------------------


def test_duration_and_record_statements_match_the_release_record(nb: dict) -> None:
    md = _markdown(nb)
    assert "not yet been recorded" not in md and "795.8 s" in md and "about 13 minutes" in md
    registry = (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    assert "not yet recorded" not in registry and "795.8 s" in registry
    record = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "is recorded yet" not in record
    manual = record[record.index("## Manual clean-runtime evidence") : record.index("## Recorded executions")]
    assert "`256fcb2` / blob `26a6d01839ff`" in manual and "795.8 s" in manual


def test_validator_refuses_the_reviewed_wording() -> None:
    validator = _load_tool("validate_release_assets")
    for stale in ("real-photo ceiling", "Hold out by caption", "its duration has not yet been recorded"):
        assert stale in validator.STALE_MARKDOWN
    assert "not a ceiling" not in "".join(validator.STALE_MARKDOWN)


# --- CPU pre-flight: the stated minimum through every stage (torch + diffusers; skipped in CI) ---------------------


def test_cpu_preflight_runs_the_smallest_stated_byod_layout_through_every_stage(preflight, tmp_path: Path) -> None:  # noqa: F811
    stages, run_root, weights = preflight
    archive = _zip(tmp_path / "byod.zip", (6, 2))  # the stated minimum plus a caption that only trains
    for stage, options in (
        ("weights", ()),
        ("prepare", ("--byod", str(archive))),
        ("encode", ()),
        ("frozen", ("--steps", "2", "--images-per-prompt", "1")),
        ("adapt", ("--epochs", "1", "--lr", "1e-3")),
        ("evaluate", ()),
        ("reload", ()),
        ("activity", ("--guidance", "1.0")),
    ):
        _run(stages, run_root, weights, stage, *options)
    report = json.loads((run_root / "outputs" / f"{stages.STEM}_evaluation_report.json").read_text(encoding="utf-8"))
    real = report["real_photo_reference"]
    assert real["n_without_reference"] == 1 and "reference_similarity" not in real  # one test photo: no 100.0
    assert report["comparison"]["reference_similarity"]["real_photo_reference"] is None
    assert set(report["comparison"]["label_accuracy"]) == {"frozen", "adapted", "real_photo_reference"}
