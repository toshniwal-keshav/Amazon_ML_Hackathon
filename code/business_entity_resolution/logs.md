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

---

## 2026-09-25 — Phase 2 · Binary Labels & Hard Negatives

**Status:** COMPLETE

### Delivered
- `src/models/__init__.py`
- `src/models/labels.py`:
  - `build_train_labels(candidates_df, ground_truth_df, pair_key_column="pair_key")` —
    returns exactly `pair_key` (string) and `y` (int8), one row per input candidate row in
    input order. Membership is a single vectorized `Series.isin` call.
  - `build_true_positive_pairs(ground_truth)` / `make_pair_key(s1, candidate)` helpers.
  - `candidate_recall(candidates_df, ground_truth_df)` returning
    `(covered, total, recall)` — the retrieval ceiling, required as a release gate by the
    master plan and therefore exposed next to the label builder.
  - Ground truth accepted in three shapes: **wide** comma-separated (the real dataset
    layout, parsed fully vectorized via `str.split` + `explode`), **long** one-row-per-pair,
    and **Mapping** `{s1_id: {candidate_id, ...}}` (the shape already used by the protected
    `metrics.py` and `splits.py`).
- `tests/test_labels.py` — 11 tests.

### Results
- `.venv/bin/python -m pytest tests/ -q` → **31 passed** (20 baseline + 11 new). No regression.
- Synthetic fixture: 179 candidate rows in, 179 label rows out; 67 positives
  (`y == 1`), 112 hard negatives (`y == 0`); `y` dtype `int8`; no duplicate `pair_key`;
  no missing labels.

### Real-data validation of the ground-truth parser
`build_true_positive_pairs` was run against the actual `train_ground_truth.tsv`
(2,206,821 rows, 123,247 of them singletons):

| check | result |
| --- | --- |
| true positive `pair_key` count | **7,638,365** — exactly matches `generate_folds.py` |
| parse time | 19.8s |
| `pair_key` format | `s1_id::candidate_id` |
| dangling keys from zero-match rows | none |

Two independent code paths (`scripts/generate_folds.py` and `src/models/labels.py`) parse the
comma-separated ground truth to the same 7,638,365 pairs, which cross-validates both.

### Decisions
- **A DataFrame ground truth must carry the correct columns even when it has no rows.**
  A frame with the right columns and zero rows legitimately means "no true pairs" and
  yields an all-negative label set. A frame with unrecognized columns raises `ValueError`
  rather than silently returning all zeros, because an all-zero label set caused by
  reading the wrong file is a catastrophic and very hard to spot failure.
- `metrics.build_ground_truth_from_tsv` remains unused: it assumes a long one-row-per-pair
  file with `entity_id` columns, which does not match the real wide ground-truth file. It
  is protected by the 20/20 baseline and was left untouched. The correct parser lives in
  `src/models/labels.py`. **This is a real trap for Person 4 — flagged here so the loader
  is not reused at inference time.**

### Next
Phase 3 — `src/models/logreg.py` `BaselineLogisticRegression`.

---

## 2026-09-25 — Phase 3 · Logistic Regression Baseline

**Status:** COMPLETE

### Delivered
- `src/models/logreg.py`:
  - `BaselineLogisticRegression` with the mandated pipeline
    `SimpleImputer(median)` → `StandardScaler()` →
    `LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)`.
  - `fit(X, y)`, `predict_proba(X) -> np.ndarray`, `predict(X)`,
    `get_feature_importance() -> pd.DataFrame`, `save`/`load` via joblib.
  - `select_feature_columns(df)` plus a `NON_FEATURE_COLUMNS` blocklist that excludes
    `pair_key`, `s1_id`, `candidate_id`, `candidate_source`, raw text, `country`,
    `fold_id`, `is_oof`, `model_version` and the score columns.
- `tests/test_logreg.py` — 14 tests.

### Results
- `.venv/bin/python -m pytest tests/ -q` → **45 passed** (20 baseline + 25 new).

### Decisions and findings
1. **Identifier columns can never reach the model.** A unique id is a perfect predictor and
   a completely useless one, so the blocklist is enforced in code and asserted by
   `test_identifier_columns_are_never_used_as_features` rather than left to caller
   discipline.
2. **A fitted column order is enforced, not assumed.** `predict_proba` reindexes a
   DataFrame to the fitted feature order. A caller passing the same columns in a different
   order gets identical predictions rather than silently scrambled coefficients. Asserted
   by `test_column_order_does_not_change_predictions`.
3. **A missing feature column at predict time raises `ValueError`**, rather than being
   treated as NaN and imputed. A renamed or dropped upstream column is a pipeline bug and
   must not be absorbed by the imputer.
4. **`inf` is rejected, not imputed.** `SimpleImputer` raises on infinity/overflow input.
   This was found by a test that initially over-claimed robustness. Infinity is a
   feature-construction bug rather than a missing value, and coercing it to NaN would hide
   the bug upstream. The behaviour is now asserted deliberately in
   `test_non_finite_input_is_rejected_loudly` so it is not "fixed" later.
5. **An all-`NaN` column is filled with `0.0` and warns.** A column entirely missing in a
   training fold has no median. This is exactly the blank-address / missing-country case
   from the master plan (Example 71's blank-address true match), so the `UserWarning` is
   deliberately left visible instead of suppressed. Documented in the module docstring.
6. `get_feature_importance` returns standardized coefficients sorted by absolute magnitude.
   They are only comparable because the scaler runs first; raw coefficients on unscaled
   features would be meaningless across columns of different units.
7. Class balance: the fixture is imbalanced (67 positives / 112 negatives) and
   `class_weight="balanced"` is required for the model to predict both classes at all
   rather than collapsing to "not a match". On the real data the equivalent collapse is
   the 0.0558 predict-nothing floor.

### Next
Phase 4 — `src/models/lightgbm_model.py` `LightGBMClassifierWrapper` with monotonic
constraints on similarity features.

---

## 2026-09-25 — Phase 4 · LightGBM + Monotonic Constraints

**Status:** COMPLETE

### Environment blocker resolved (no root, no system modification)
`lightgbm` was absent. `pip install lightgbm` succeeded (4.7.0, prebuilt manylinux wheel,
authorized by the master directive), but importing it then failed:

```
OSError: libgomp.so.1: cannot open shared object file: No such file or directory
```

LightGBM's wheel links against the GNU OpenMP runtime, which this WSL image does not ship.
`apt-get install libgomp1` was **not** an option: the session runs as uid 1000, so there is
no root. Resolution, in order of preference:

1. `scripts/fetch_opensmp_runtime.sh` downloads the official Ubuntu 24.04 `libgomp1`
   package (`libgomp1_14.2.0-4ubuntu2~24.04.1_amd64.deb`) and extracts it with
   `dpkg-deb -x` into `.vendor/libgomp/` — no root, no system change.
2. The extracted `libgomp.so.1.0.0` (344 KB) is **committed** to the repo, so
   `pytest tests/ -v` works on a fresh clone with no system changes at all.
3. `_import_lightgbm()` in `src/models/lightgbm_model.py` tries the **normal import first**,
   so a host with a system OpenMP runtime always uses its own copy. Only on `OSError` does
   it `ctypes.CDLL(..., RTLD_GLOBAL)` the vendored library, purge the half-initialised
   `lightgbm` modules from `sys.modules`, and retry.

Verified end to end: `lightgbm 4.7.0` imports and trains with **no `LD_LIBRARY_PATH` and no
environment variable of any kind**, and the fetch script is idempotent on re-run.

### Delivered
- `src/models/lightgbm_model.py`:
  - `LightGBMClassifierWrapper` — `binary` / `binary_logloss`, early stopping on validation
    average-precision or logloss, native NaN handling, joblib `save`/`load`.
  - `MONOTONIC_SIMILARITY_COLUMNS` = `name_char_cos`, `name_token_set`, `name_token_sort`,
    `name_contain_idf`, `addr_char_cos`, `addr_contain_idf`, `addr_numeric_jaccard`,
    `best_score`, each constrained to `+1` via `monotone_constraints_method="advanced"`.
  - `build_monotonic_constraints()` / `constrained_feature_names()` /
    `get_monotonic_constraints()` to report the constraint actually applied.
  - `select_feature_columns()` with the identifier/metadata blocklist.
- `tests/test_lightgbm.py` — 17 tests.
- `scripts/fetch_opensmp_runtime.sh`.
- `code/business_entity_resolution/requirements.txt` — was empty; now pins the verified
  versions (P3-relevant subset, additive only).

### Results
- `.venv/bin/python -m pytest tests/ -q` → **62 passed** (20 baseline + 42 new).

### Decisions and findings
1. **Monotonic constraints are on by default.** An initial bug made `monotone_constraints`
   default to *no* constraints, which silently defeated the whole point of the phase. The
   default is now `MONOTONIC_SIMILARITY_COLUMNS`; `monotone_constraints=[]` is the
   explicit opt-out used only by the control test.
2. **Only genuine similarity columns are constrained.** `addr_numeric_conflict`,
   `name_len_diff` and `n_routes` are deliberately left free: "more routes" or "larger
   length difference" is not monotonically related to matching, and forcing a direction on
   them would inject a false prior.
3. **The monotonicity test bug was mine, not LightGBM's.** The test initially reported a
   catastrophic -0.997 probability drop on all 8 constrained features. Diagnosis: the
   helper concatenated all reference-row curves into one array and then called `np.diff`
   on the flattened result, so `np.diff` compared the *last* point of one curve (grid
   maximum) against the *first* point of the next curve (grid minimum), inventing a fake
   decrease at the seam. Fixed by stacking curves into a `(n_rows, n_points)` array and
   taking `np.diff(..., axis=1)`. The real per-curve curves were flat or increasing
   throughout, confirming the constraints hold. This was verified by an independent
   standalone reproduction before the test was touched.
4. **The monotonicity test is guarded by a control test.**
   `test_unconstrained_model_violates_monotonicity_on_same_data` trains the same model
   without constraints and asserts its worst drop is *worse* than the constrained model's.
   Without this, a vacuously monotone model would let the main test pass for the wrong
   reason. The control passes, so the constraint is demonstrably doing work.
5. **LightGBM 4.7 renamed the evaluation arguments.** `eval_set` now emits
   `LGBMDeprecationWarning`; `eval_X`/`eval_y` are the supported spellings and are used
   instead. `eval_X` rejects a list of DataFrames, so a single validation set is passed
   unwrapped. The test suite is now warning-clean apart from the two deliberate
   all-`NaN`-column imputer warnings.
6. **Native NaN handling, no imputation.** Required by the master plan: Example 71 is a
   true match with a blank address, and a blank must be learned from other evidence
   rather than median-imputed into a mid-range similarity. Asserted by
   `test_native_nan_support_for_missing_address_and_country`.

### Next
Phase 5 — `src/models/train_cv.py` 5-fold CV runner and OOF `scores.parquet`.

---

## 2026-09-25 — Phase 5 · 5-Fold CV Runner & OOF Scores

**Status:** COMPLETE

### Delivered
- `src/models/train_cv.py`:
  - `train_cv(features_df, labels_df, folds_df, model_cls, model_params, n_folds=5,
    model_version="model_v1", expected_folds=None) -> tuple[pd.DataFrame, list]`
  - `prepare_training_frame(features_df, labels_df, folds_df)` — validated join producing
    one row per candidate pair carrying `y` and `fold_id`.
  - `SCORES_COLUMNS` / `SCORES_ARROW_SCHEMA` / `write_scores_parquet(scores_df, path)`.
  - `coverage_report(scores_df, frame)` for run logging.
- `tests/test_train_cv.py` — 11 tests.

### Results
- `.venv/bin/python -m pytest tests/ -q` → **73 passed** (20 baseline + 53 new).
- Synthetic run: 179 candidate rows in → 179 OOF score rows out, one per `pair_key`, all
  `is_oof=True`, spread over 5 folds; each model trained on exactly the other 4 folds.

### Decisions and findings
1. **Leakage is structurally impossible, not merely tested for.** Folds are joined at the
   S1 entity level, so a pair's fold is a property of its entity. The training frame for
   fold `k` is built from a boolean mask excluding fold `k`, and the test asserts that
   fold `k` is absent from its own training fold set and that exactly 4 folds are present.
   No later feature or join step can reintroduce the held-out rows.
2. **A fold coverage gap raises instead of silently shrinking the training set.** If an S1
   entity in the candidate set has no `fold_id`, `train_cv` raises naming the entity count.
   A silent drop here would quietly remove training data and inflate every score computed
   from it. Covered by `test_every_s1_entity_present_in_folds_or_raises`.
3. **pandas 3.0 would have broken the `scores.parquet` contract.** pandas 3.x stores text
   columns as `StringDtype`, which pyarrow maps to Arrow `large_string` (64-bit offsets),
   not the contracted `string` (32-bit). A bare `pa.Table.from_pandas(scores)` therefore
   produces an artifact that disagrees with `docs/schemas.md`, and Person 4's loader and
   validator would end up written against a schema nobody produces. Fixed by enforcing the
   schema at the write boundary: `SCORES_ARROW_SCHEMA` + `write_scores_parquet()`, with a
   test asserting the written file's schema field-for-field. Noted in `docs/schemas.md`.
4. **`p_cal` currently equals `p_raw` and is documented as such.** No calibrator is fitted
   yet; writing a pretended calibrated column would misrepresent the artifact. Fitting an
   isotonic/sigmoid calibrator on OOF predictions is the natural follow-up and belongs in
   this module so it is only ever fitted on OOF data.
5. **`expected_folds` parameter added** so Phase 7's LOCO harness can reuse this exact
   leak-free splitting logic over country slices instead of reimplementing it.
6. `random_state=42` is injected only when the model class accepts it, so every seeded
   model stays reproducible without breaking models that do not take the argument.
7. Test dtype assertions were corrected from `object` to `pd.api.types.is_string_dtype(...)`
   for pandas 3.x, which was a test bug rather than an implementation problem.

### Next
Phase 6 — `src/decision/threshold.py` global and two-threshold tuning, decision rules, and
`artifacts/decision_config.json`.
