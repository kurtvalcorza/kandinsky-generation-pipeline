"""The weight prose may only quote SHA-256 digests and byte counts that a committed manifest records."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_release_assets", ROOT / "tools" / "validate_release_assets.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = _load_validator()


def _copy_docs(tmp_path: Path) -> Path:
    for name in validator.WEIGHT_DOCS:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, tmp_path / name)
    for manifest in ROOT.glob("weights/*/dimer-base-manifest.json"):
        target = tmp_path / manifest.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(manifest, target)
    return tmp_path


def test_repository_weight_facts_match_manifests():
    validator.validate_weight_facts()


def test_wrong_byte_count_is_rejected(tmp_path):
    root = _copy_docs(tmp_path)
    card = root / "MODEL_CARD.md"
    card.write_text(card.read_text(encoding="utf-8") + "\n- stray (5,012,319,952 bytes)\n", encoding="utf-8")
    with pytest.raises(validator.ValidationError, match="byte counts"):
        validator.validate_weight_facts(root)


def test_wrong_digest_is_rejected(tmp_path):
    root = _copy_docs(tmp_path)
    weights = root / "docs" / "WEIGHTS.md"
    weights.write_text(weights.read_text(encoding="utf-8") + "\nSHA-256: `" + "ab" * 32 + "`\n", encoding="utf-8")
    with pytest.raises(validator.ValidationError, match="SHA-256"):
        validator.validate_weight_facts(root)
