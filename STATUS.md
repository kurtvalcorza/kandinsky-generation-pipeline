# Release status

Current status: **Release-grade** — at `b674640` the `E2E` tutorial notebook `tutorials/kandinsky_generation_colab.ipynb` passed a clean-runtime `Run all` of the default path on a Kaggle T4 and the REL12 BYOD journey (representative photographs accepted through `BYOD_PATH` and carried through adaptation, evaluation, export and reload; two incompatible inputs refused), both on 2026-09-29 and recorded in `docs/release-verification.md`.

One deployment gate applies: the served weights set is approximately 15.9 GB (decoder ~5.28 GB + shared prior ~10.57 GB), falling under the fleet's >9 GB publication convenience gate. Following the PixArt-Σ / Toto precedent, the gate is waived and weights provenance is recorded in `docs/WEIGHTS.md` and `MODEL_CARD.md`.
