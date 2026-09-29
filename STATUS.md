# Release status

Current status: **Candidate** — a clean-runtime `Run all` of the `E2E` tutorial notebook `tutorials/kandinsky_generation_colab.ipynb` at `6347e25` passed on a Kaggle T4 on 2026-09-29, with 12/12 code cells, fine-tuning, evaluation, export and an exact reload (`docs/release-verification.md`). Promotion to **Release-grade** still needs BYOD evidence (REL12): the BYOD branch has no location field yet, so it can only be exercised through the Colab upload dialog.

One deployment gate applies: the served weights set is approximately 15.9 GB (decoder ~5.28 GB + shared prior ~10.57 GB), falling under the fleet's >9 GB publication convenience gate. Following the PixArt-Σ / Toto precedent, the gate is waived and weights provenance is recorded in `docs/WEIGHTS.md` and `MODEL_CARD.md`.
