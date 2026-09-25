# Person 3 Execution Log — ML, Validation, Models & Decision Pipeline

Owner: Person 3 (ML / Validation / Models / Decision)
Branch: `person3-ml`
Rollback anchor: commit `5872636` (verified baseline, 20/20 tests)
Log format: one entry per phase start/end, blocker, and decision.

---

## 2026-09-25 — Phase 0 · Environment & Repository Audit

**Status:** COMPLETE

### Verified
- Interpreter: `/mnt/d/PROJECTS/Amazon_ML_Hackathon/.venv/bin/python` (Python 3.12.3).
  Fresh shells do not have `python` on PATH — always use the explicit interpreter path.
- Installed: numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, pyarrow 25.0.1,
  scipy 1.18.1, pytest 9.1.1, joblib.
- Not installed: lightgbm. Deferred to Phase 4 per master directive sequencing.
- `python: command not found` was a PATH issue, not a missing dependency. Resolved by
  using the explicit interpreter for every command.
- Git: `git status --short` clean, `git log -2 --oneline` → `5872636` (HEAD),
  `6657997` (parent). No push, no reset, no branch deletion performed.

### Repository layout (verified)
- Working P3 validation code + 20-test baseline: root `src/validation/`, root `tests/`.
- `code/business_entity_resolution/src/` contains the P3 scaffold; the P3 target
  modules (`models/labels.py`, `models/logreg.py`, `models/lightgbm_model.py`,
  `models/train_cv.py`, `decision/threshold.py`, `validation/loco.py`) are empty
  placeholders awaiting implementation in Phases 2–7.
- `artifacts/` exists and is empty. `scripts/` and `docs/` do not exist yet.

### Data headers (verified)
- `dataset/train/train_source1.tsv`: `entity_id`, `business_name`, `business_address`, `country`
- `dataset/train/train_source2.tsv`, `train_source3.tsv`: same schema as source1
- `dataset/train/train_ground_truth.tsv`: `source1_entity_id`, `matched_entity_ids`
  (comma-separated; empty field = true singleton / zero-match entity)
- Row counts: `train_source1.tsv` = 2,206,822 lines (1 header + 2,206,821 S1 entities);
  `train_ground_truth.tsv` = 2,206,822 lines (1 header + one row per S1 entity).

### Decisions
1. **Preserve the 20/20 baseline.** `src/validation/metrics.py` and
   `src/validation/splits.py` are committed at `5872636` and must not be edited.
   Verified: `.venv/bin/python -m pytest tests/ -q` → `20 passed`.
2. **New P3 modules are created in the committed location, tests stay in root `tests/`.**
   The master directive's roadmap paths (`src/models/labels.py`, `src/decision/threshold.py`,
   `src/validation/loco.py`, `tests/conftest.py`, `scripts/generate_folds.py`,
   `artifacts/folds.parquet`) are interpreted relative to the repository root, which is where
   the 20/20 baseline already lives. This keeps a single import root (`src.*`) so the
   existing tests and new code share one namespace and no duplicate competing
   implementations are created. The empty `code/business_entity_resolution/src/` scaffold is
   left untouched as the assembly target for final packaging.
3. **`docs/schemas.md` does not exist in the repository.** The artifact column contracts
   named by the master directive (Phase 1 `folds.parquet`, Phase 2 `train_labels.parquet`,
   Phase 5 `scores.parquet`) are therefore treated as canonical and implemented verbatim as
   specified. `docs/schemas.md` will be created at the end of Phase 1 to record the contracts
   actually in force, so later phases reference a real document rather than an absent one.
4. **Ground-truth parsing must be column-driven, not via `metrics.build_ground_truth_from_tsv`.**
   The existing `build_ground_truth_from_tsv` in the protected `metrics.py` assumes a
   long-format one-row-per-pair file with `entity_id` columns, which does **not** match the
   actual wide/comma-separated `train_ground_truth.tsv` format. It is left unmodified
   (baseline protection) and is not used by the production ingestion path. Phase 1 implements
   correct parsing in `scripts/generate_folds.py`.
5. **Memory safety:** all TSV reads use `dtype=str, keep_default_na=False`, explicit `usecols`,
   and chunked iteration. No full multi-GB table loads.

### Open items / blockers
- P1 `candidates.parquet` and P2 `features.parquet` are absent. Phases 2–7 are therefore
  developed and tested against synthetic fixtures. This validates mechanics and
  reproducibility only — **no synthetic result may be reported as a competition-valid
  score or model baseline.**

---

## 2026-09-25 — Phase 1 · Frozen Folds + Synthetic Fixtures

**Status:** COMPLETE

### Delivered
- `scripts/generate_folds.py` — streams `train_source1.tsv` (`entity_id`, `country` only)
  and `train_ground_truth.tsv` (`source1_entity_id`, `matched_entity_ids` only), both with
  `dtype=str, keep_default_na=False`, `usecols` and `chunksize=200_000`; assigns folds via
  the protected `assign_folds_from_ground_truth(..., n_folds=5, random_state=42)`; writes
  via the protected `folds_to_parquet()`. Country values are interned and duplicate S1 ids
  are dropped. No full-table load.
- `tests/conftest.py` — `synthetic_s1_pool` (50 S1 entities), `synthetic_candidates_df`
  (7 contract columns), `synthetic_ground_truth`, `synthetic_features_df` (35 contract
  columns), plus a `synthetic_folds_df` helper for later phases.
- `docs/schemas.md` — created; records the canonical contracts now in force for
  `folds.parquet`, `candidates.parquet`, `train_labels.parquet`, `features.parquet`,
  `scores.parquet`, `decision_config.json`, and the evaluation contract.
- `artifacts/folds.parquet` — 15.3 MB, generated and frozen.

### Results (real training data)
Command: `.venv/bin/python scripts/generate_folds.py` (36.0s)

| quantity | value |
| --- | --- |
| S1 entities read | 2,206,821 |
| S1 entities assigned | 2,206,821 (100.00%) |
| S1 entities with no ground-truth row | 0 |
| zero-match (singleton) S1 entities | 123,247 |
| total true match pairs | 7,638,365 |
| parquet schema | `s1_id: string`, `fold_id: int8` |
| duplicate `s1_id` values | 0 |
| `fold_id` values outside `[0, 5)` | 0 |
| relative fold imbalance | 0.0000% |

Fold sizes: 441,365 / 441,364 / 441,364 / 441,364 / 441,364 (each 20.00%).

The zero-match count of **123,247 exactly reproduces the documented statistic** in
`planning/dataset_analysis_results.md`, which independently confirms the comma-separated
`matched_entity_ids` parsing is correct.

### Stratification quality (real data)
Composite strata `country x has_match x match_count_bucket` → 8 populated strata
(train contains only `US` and `India`; `France` is test-only).

| stratum | overall share | per-fold share | max abs deviation |
| --- | --- | --- | --- |
| US, has_match, 4+ | 28.724% | 28.724% in every fold | 0.000149 pp |
| US, has_match, 2-3 | 24.658% | 24.658% in every fold | 0.000192 pp |
| India, has_match, 4+ | 19.235% | 19.235% in every fold | 0.000145 pp |
| India, has_match, 2-3 | 16.399% | 16.399% in every fold | 0.000143 pp |
| US, zero-match | 3.349% | 3.348–3.349% | 0.000183 pp |
| US, has_match, 1 | 3.249% | 3.248–3.249% | 0.000180 pp |
| India, zero-match | 2.236% | 2.236% | 0.000177 pp |
| India, has_match, 1 | 2.151% | 2.151% | 0.000140 pp |

Max stratum-share deviation across all folds: **0.000192 pp** (integer rounding only).
Zero-match share is 5.585% in every fold, matching the overall 5.585%.

Analysis correction: an earlier ad-hoc check reported a 0.1785 stratum deviation. That
metric normalized per-stratum across folds and then compared it against each stratum's
share of the full dataset, mixing two different denominators. The corrected
within-fold comparison above is the valid one; the folds themselves were never affected.

### Reproducibility
`artifacts/folds.parquet` MD5 `773f8020855f99edb2bcebc99e581ddd`, byte-identical across two
independent full runs of the script. The artifact is deterministic for
`random_state=42`.

### Fixture validation
- 50 S1 entities: 17 US / 17 India / 16 France.
- 67 true pairs; match-count buckets populated: 0→19, 1→13, 2→9, 3→3, 4→3, 5→3.
- 179 candidate pairs; **true-pair candidate recall 100% (67/67)**; no duplicate `pair_key`.
- 19 zero-match S1 entities have only negative candidates, so the Phase 6 decision layer
  has genuine abstention cases to test against.
- 51 deliberate NaNs across address similarity and `country_eq` so NaN handling is
  exercised by the Phase 3/4 model tests.

### Verification
- `.venv/bin/python -m pytest tests/ -v` → **20 passed**. Baseline intact, no regression.
- `.venv/bin/python -m compileall -q src tests scripts` → clean.

### Decisions
- `artifacts/folds.parquet` is now the frozen canonical fold assignment. Every later phase
  must read it and must never re-derive folds.
- Candidate pairs inherit their S1 entity's fold, so Phase 5 leakage is structurally
  impossible rather than merely tested for.
- LOCO in Phase 7 is limited to US↔India because those are the only countries present in
  training. France is test-only and cannot appear in a training-side diagnostic.

### Blockers
- None for Phases 2–7, which proceed on synthetic fixtures.
- Real `candidates.parquet` (P1) and `features.parquet` (P2) remain absent, so no
  competition-valid model score can be produced yet.

### Next
Phase 2 — `src/models/labels.py` binary label and hard-negative builder.
