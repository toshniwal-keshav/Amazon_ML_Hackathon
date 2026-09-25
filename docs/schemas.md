# Shared Artifact Schemas

This document records the shared artifact contracts used by P1, P2, P3, and P4 during development.

The final submission contracts are reproduced in the final `README.md`.

---

## 1. Global Rules

- All entity IDs are strings and must never be inferred as numeric values.

- `pair_key` is defined as:

  `s1_id + "::" + candidate_id`

  wherever an artifact represents an S1-candidate pair.

- The official challenge input files are TSV files and must be read with an explicit tab separator.

- The official TSV files do not contain a separate `source` column. Source is available from the filename and/or entity-ID prefix (`S1-`, `S2-`, `S3-`). The internal `records.parquet` artifact may materialize a `source` column for convenience.

- Training contains US and India. Test additionally contains France.

- `country` must be treated as an open set of string labels.

- The pipeline must not hard-code, filter, or one-hot the country universe to `{US, India}`.

- Every test Source 1 entity, including France, must appear in `matching_results.tsv`.

### Leakage and candidate-generation rules

- Candidate generation, normalization, and feature definitions must not use label truth.

- `train_ground_truth.tsv` may only be consumed by training/evaluation logic.

- No feature or route may be derived from `train_ground_truth.tsv` match counts, match frequency, or which pairs are positive.

- Frequency features must come from raw text or candidate appearance, never from label truth.

- No full dense S1 × candidate-pool matrices may be created.

- `candidate_pairs.tsv` must represent the final candidate set actually passed to the final matching model for inference.

---

## 2. Shared Artifact Contracts

Every pairwise artifact uses `pair_key = s1_id + "::" + candidate_id` as a string.

All ID columns are strings.

| Artifact | Owner | Columns | Consumed by | Must exist | Frozen at |

|---|---|---|---|---|---|

| `records.parquet` | P2 | `entity_id, source, raw_name, raw_address, raw_country, norm_name_cons, norm_name_aggr, norm_address_cons, country_norm` | P1, P2, P3 (via features) | Hour 4 (subsample), Hour 8 (full) | Normalization logic: Hour 24 |

| `folds.parquet` | P3 | `s1_id, fold_id` | P3, P4 | Hour 3 | Hour 3 — never touched again |

| `retrieval_events.parquet` | P1 | `pair_key, s1_id, candidate_id, candidate_source, route, rank, score` | P2, P4 | Hour 4 (subsample), Hour 10 (full train), Hour 20 (full test) | Route set: Hour 12 |

| `candidates.parquet` | P1 | `pair_key, s1_id, candidate_id, candidate_source, n_routes, best_rank, best_score` | P3, P4 | Same as `retrieval_events.parquet` | Same as `retrieval_events.parquet` |

| `features.parquet` | P2 | `pair_key, s1_id, candidate_id` plus name/address/country/missingness/retrieval features | P3 | Hour 5 (subsample), Hour 11 (full train), Hour 22 (full test) | Feature list: Hour 24 |

| `train_labels.parquet` | P3 | `pair_key, y` | P3 | Hour 6 | Derived fresh whenever candidates change, up to Hour 12 |

| `scores.parquet` | P3 | `pair_key, s1_id, candidate_id, p_raw, p_cal, fold_id, is_oof, model_version` | P3 decision layer, P4 | Hour 12 (LR), Hour 20 (LightGBM) | Model/hyperparameters: Hour 36 |

| `matching_results.tsv` | P4 | `source1_entity_id, matched_entity_ids` | Validator, submission | First real pair: Hour 20 | Pipeline code: Hour 48 |

| `candidate_pairs.tsv` | P4 | `source1_entity_id, candidate_entity_ids` | Validator, submission | First real pair: Hour 20 | Pipeline code: Hour 48 |

| `experiment_tracker.csv` | P4 / everyone writes rows | `experiment_id, hypothesis, change, metric, candidate_recall, avg_candidates, runtime, result, decision, next_action` | Everyone | Hour 3, updated continuously | Never frozen |

| `logs/person{1-4}.md` | Each person | Free-form timestamped entries | P4 | Hour 0, updated continuously | Never frozen |

---

## 3. `records.parquet`

### Owner

P2.

### Purpose

Shared normalized record table used by downstream candidate generation, feature engineering, and model work.

### Columns

```text

entity_id

source

raw_name

raw_address

raw_country

norm_name_cons

norm_name_aggr

norm_address_cons

country_norm

````

### Rules

* `entity_id` is a string.

* `source` is an internal convenience column. It is not a separate column in the official input TSV files.

* Normalization logic is frozen at Hour 24.

---

## 4. `folds.parquet`

### Owner

P3.

### Columns

```text

s1_id

fold_id

```

### Rules

* `s1_id` is a string.

* The fold split is frozen at Hour 3 and must not be changed afterward.

---

## 5. `retrieval_events.parquet`

### Owner

P1.

### Purpose

Stores individual retrieval events produced by candidate-generation routes.

### Columns

```text

pair_key

s1_id

candidate_id

candidate_source

route

rank

score

```

### Rules

* `pair_key`, `s1_id`, and `candidate_id` are strings.

* Each retrieval route contributes retrieval events.

* The route set is frozen at Hour 12.

* P2 may pivot these retrieval events into retrieval-derived features.

* P4 uses the final candidate information to construct `candidate_pairs.tsv`.

---

## 6. `candidates.parquet`

### Owner

P1.

### Purpose

Aggregated candidate view derived from `retrieval_events.parquet`.

### Columns

```text

pair_key

s1_id

candidate_id

candidate_source

n_routes

best_rank

best_score

```

### Rules

* `pair_key`, `s1_id`, and `candidate_id` are strings.

* This is the canonical candidate table for downstream recall evaluation and integration.

* `candidate_pairs.tsv` must ultimately represent the final candidate set used for model inference.

---

## 7. `features.parquet`

### Owner

P2.

### Required identity columns

```text

pair_key

s1_id

candidate_id

```

### Additional feature columns

The artifact also contains the name, address, country, missingness, and retrieval features defined by P2's feature contract.

The feature list and exact definitions are frozen at Hour 24.

### Rules

* All identity columns are strings.

* Feature definitions must not use label truth.

* No new feature columns may be introduced after the Hour 24 freeze without a team-wide announcement.

* Retrieval-derived features may be created from `retrieval_events.parquet`.

---

## 8. `train_labels.parquet`

### Owner

P3.

### Purpose

Training labels for candidate pairs.

### Columns

```text

pair_key

y

```

### Construction

`train_labels.parquet` is built by joining `candidates.parquet` against the official `train_ground_truth.tsv`.

### Rules

* Label truth may be used for constructing training labels.

* Label truth must not be used to create candidate-generation, normalization, or feature definitions.

* This artifact is derived rather than permanently frozen because candidate changes can require regeneration up to Hour 12.

---

## 9. `scores.parquet`

### Owner

P3.

### Purpose

Stores model scores used by the decision layer and consumed by P4 during inference integration.

### Columns

```text

pair_key

s1_id

candidate_id

p_raw

p_cal

fold_id

is_oof

model_version

```

### Rules

* `pair_key`, `s1_id`, and `candidate_id` are strings.

* `is_oof` must be `True` for any score used in threshold tuning or the reverse-consistency rule.

* Scores compared to each other in reverse-consistency must both be OOF scores.

* The model and hyperparameters are frozen at Hour 36.

---

## 10. Official Output: `output/matching_results.tsv`

### Columns

```text

source1_entity_id

matched_entity_ids

```

### Format

* Tab-separated columns.

* Exactly one row for every test Source 1 entity.

* `matched_entity_ids` is a comma-separated list of predicted matching IDs.

* There must be no quoting inside the comma-separated ID list.

* There must be no duplicate IDs inside one list.

* Only Source 2 and Source 3 IDs that actually exist in the test set may appear as predictions.

* An S1 ID must never appear as a prediction.

* When there is no predicted match, `matched_entity_ids` must be empty.

* Every test Source 1 entity must appear exactly once.

Example:

```text

source1_entity_id   matched_entity_ids

S1-00001    S2-00047,S2-00193,S3-00812

S1-00002    S3-00004

S1-00003    

```

---

## 11. Official Output: `output/candidate_pairs.tsv`

### Columns

```text

source1_entity_id

candidate_entity_ids

```

### Format

* Tab-separated columns.

* Exactly one row for every test Source 1 entity.

* `candidate_entity_ids` is a comma-separated list.

* There must be no duplicate candidate IDs inside one list.

* Only test Source 2 and Source 3 IDs may appear.

* An empty candidate list is allowed.

* Every ID appearing in `matching_results.tsv` must also appear in the corresponding `candidate_pairs.tsv` candidate list.

### Critical meaning of this file

`candidate_pairs.tsv` is ****not**** an earlier blocking output.

It must contain the ****final candidate set actually passed to the final matching model for inference****.

If the pipeline has several candidate-generation or filtering stages, this file represents the last stage immediately before model inference.

---

## 12. Critical Candidate / Prediction Consistency Invariant

The exact same final candidate DataFrame must be used for both:

```text

Final candidate DataFrame

        |

        +----------------------+

        |                      |

        v                      v

P2 feature generation      candidate_pairs.tsv

        |

        v

P3 model scoring / decision

        |

        v

matching_results.tsv

```

Therefore:

* `candidate_pairs.tsv` must be generated from the exact same final candidate DataFrame used for model inference.

* `candidate_pairs.tsv` must not be independently regenerated after scoring.

* Every final prediction must be a member of the corresponding final candidate set.

* A prediction that is absent from `candidate_pairs.tsv` is a pipeline inconsistency.

---

## 13. Hard Validation Rules for Official Outputs

Before submission, the following conditions must hold:

```text

number of rows in matching_results.tsv

    =

number of test Source 1 entities

```

```text

number of rows in candidate_pairs.tsv

    =

number of test Source 1 entities

```

The set of Source 1 IDs in both output files must equal the set of test Source 1 entity IDs.

For every Source 1 entity:

```text

predicted IDs ⊆ candidate IDs

```

Also:

```text

predicted IDs ∩ S1 IDs = empty set

```

Predictions and candidates may contain only IDs belonging to test Source 2 or Source 3.

---

## 14. Non-Negotiable Score / Validation Rules

* `is_oof` must be `True` for any score used in threshold tuning.

* `is_oof` must be `True` for scores used by the reverse-consistency rule.

* Reverse-consistency comparisons must use comparable OOF scores.

* Threshold tuning must not use ordinary in-sample training predictions.

* Test labels are unavailable.

* Model selection, thresholding, and decision-rule tuning must use leakage-safe training validation.

* Leaderboard results must not be used to tune thresholds, features, hyperparameters, or candidate-generation configurations.

---

## 15. Schema Freeze Schedule

### Hour 12

Freeze:

* Candidate-generation route set, including route parameters such as k and thresholds.

* `docs/schemas.md` as-built during development.

* Confirm that `folds.parquet` remains unchanged.

### Hour 24

Freeze:

* Feature list.

* Exact feature definitions.

* Final blocking parameters such as k values and block-size caps.

No new feature columns should be introduced after this point without a team-wide announcement.

### Hour 36

Freeze:

* Model type.

* Hyperparameter search space.

* Decision-layer rule set.

After this point, no further hyperparameter exploration should be started; only refits with the chosen configuration are performed.

### Hour 48

Freeze:

* Entire pipeline codebase.

From this point onward, only bug fixes are allowed.

The final model is retrained on full training data and real test predictions are generated from the frozen pipeline.

---

## 16. Development vs Final Submission

`docs/schemas.md` is development-only documentation.

The final submission must contain the contracts necessary to reproduce the outputs in:

```text

code/business_entity_resolution/README.md

```

The final ZIP has a separate required structure and does not automatically include development-only files such as `docs/`, `logs/`, `experiments/`, or temporary artifacts.

---

## 17. Shared Contract Change Rule

Shared artifact columns and data types are coordination-sensitive.

Any change to a shared contract must be communicated to the team before downstream code is updated, because P1, P2, P3, and P4 depend on these interfaces.

The shared contracts covered here are:

```text

records.parquet

folds.parquet

retrieval_events.parquet

candidates.parquet

features.parquet

train_labels.parquet

scores.parquet

matching_results.tsv

candidate_pairs.tsv

```
