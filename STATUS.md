# Release status

Current status: **Release-grade** — at `256fcb2` the `E2E` tutorial notebook `tutorials/kandinsky_generation_colab.ipynb` runs its stages in an isolated hash-locked environment and installs nothing into the kernel. It passed `Run all` in one pass on Google Colab (T4, 2026-09-30) and on a clean Kaggle T4 in strict single-pass mode, and the REL12 BYOD journey on Kaggle (2026-09-29); all recorded in `docs/release-verification.md`.

One deployment gate applies: the served weights set is approximately 15.9 GB (decoder ~5.28 GB + shared prior ~10.57 GB), falling under the fleet's >9 GB publication convenience gate. Following the PixArt-Σ / Toto precedent, the gate is waived and weights provenance is recorded in `docs/WEIGHTS.md` and `MODEL_CARD.md`.
