# PERSON 2 LOG — Preprocessing, Normalization & Feature Engineering

**Owner:** Person 2 (Normalization & Feature Engineering)  
**Role Context:** Amazon ML Challenge 2026: Business Entity Resolution  
**Status:** Complete, Verified, Ready for P3/P4 Handoff  

---

## 1. Scope and Mission Summary
Person 2 owns the descriptive representation of entity records and candidate pairs:
1. **Preprocessing / Normalization (`src/preprocessing/`)**: Transform raw, noisy business names, addresses, and country codes into standardized, multilingual-safe representations.
2. **Corpus Statistics Mining (`src/preprocessing/corpus_stats.py`)**: Extract label-free high-frequency trailing legal suffixes (including French `SARL`, `SAS`, `EURL` for test distribution shift, Indian `Pvt Ltd`, and US `LLC`/`Inc`) and leading DBA/trade markers.
3. **Pairwise Feature Engineering (`src/features/`)**: Compute leak-free similarity metrics across name, address, country, missingness, and retrieval provenance for every candidate pair in `candidates.parquet`.
4. **Testing & Invariants (`tests/`)**: 100% unit-tested against real evidence cases from `training_match_examples.md`.

---

## 2. Canonical Contracts & Schemas

### A. Preprocessing Output (`records.parquet`)
| Column | Type | Description |
|---|---|---|
| `entity_id` | `string` | Unique entity identifier (e.g., `"S1-12345"`) |
| `source` | `string` | Inferred source (`"S1"`, `"S2"`, or `"S3"`) |
| `raw_name` | `string` | Original unnormalized business name |
| `raw_address` | `string` | Original unnormalized address |
| `raw_country` | `string` | Original raw country string |
| `norm_name_cons` | `string` | Conservative normalized name (Unicode NFKC, casefolded, punctuation-stripped, accents folded) |
| `norm_name_aggr` | `string` | Aggressive normalized name (stripped leading DBA markers and trailing mined legal suffixes) |
| `norm_address_cons` | `string` | Conservative normalized address |
| `country_norm` | `string` | Standardized country code (uppercase, stripped, empty if missing) |

### B. Pairwise Features Output (`features.parquet`)
| Column Group | Feature Names | Dtype / Null Semantics |
|---|---|---|
| **Identifiers** | `pair_key` (`"s1_id::cand_id"`), `s1_id`, `candidate_id` | `string` |
| **Name Features** | `name_exact_cons`, `name_exact_aggr`, `name_char_cos`, `name_token_sort_ratio`, `name_token_set_ratio`, `name_containment_s1_in_cand`, `name_containment_cand_in_s1`, `name_len_diff`, `name_len_ratio`, `name_token_count_diff`, `name_first_token_match` | `float32` |
| **Address Features** | `addr_exact_cons`, `addr_char_cos`, `addr_token_sort_ratio`, `addr_containment_s1_in_cand`, `addr_containment_cand_in_s1`, `addr_len_diff`, `addr_num_jaccard`, `addr_num_conflict`, `addr_first_num_match` | `float32` (Missing addresses evaluate to `np.nan` on similarity metrics) |
| **Country & Missingness** | `country_eq`, `name_missing_cand`, `addr_missing_s1`, `addr_missing_cand`, `addr_missing_either`, `min_name_addr_sim`, `max_name_addr_sim`, `both_strong_match` | `float32` (`country_eq` is `np.nan` if either country missing) |
| **Retrieval Metadata** | `n_routes`, `retrieval_best_rank`, `retrieval_best_score`, `retrieved_by_exact_name`, `retrieved_by_tfidf_name`, `retrieved_by_tfidf_address`, `retrieved_by_numeric`, `retrieved_by_reverse` | `float32` (`np.nan` if route not triggered) |

---

## 3. Evidence-Based Guardrails Handled

1. **The Kalyani Case (Example 66 - Subset Matching):**
   * Symmetric similarities between `"Kalyani Welfare Society"` and `"Kalyani"` are ~0.4, but candidate token containment in S1 is `1.0`.
   * Implemented directional containment: `name_containment_cand_in_s1` and `name_containment_s1_in_cand`.
2. **The NEXGILD Case (Example 69 - Address-Driven Match):**
   * `"Oncology Associates"` vs `"NEXGILD"` shares identical address with zero name similarity.
   * Address features produce `addr_char_cos > 0.95` and `addr_first_num_match = 1.0` while name similarity gracefully sits at `< 0.20` without throwing errors.
3. **The Blank Address Case (Example 71 - Missing Evidence $\neq$ Negative Evidence):**
   * True match with missing candidate address produces `addr_missing_cand = 1.0` and `addr_char_cos = np.nan`, `addr_num_jaccard = np.nan`.
   * Preserves `NaN` for GBDT native tree-split handling without zero-filling.
4. **The Digit Typo Case (Example 71 - Soft Conflict):**
   * `"703 Beacon Court"` vs `"70 Beacon Court"` computes `addr_num_conflict = 1.0` and `addr_num_jaccard = 0.0`, while `addr_char_cos > 0.70` captures street similarity.
5. **Multilingual & Mixed Script (Example 68 - Damani):**
   * Mixed Devanagari (`दिल्ली`) and Latin script records: normalization uses Unicode category filtering (`L*`, `M*`, `N*`), guaranteeing that Hindi matras/vowel signs are preserved without ASCII corruption.
6. **French Unseen Distribution Shift:**
   * Dynamic corpus statistics mining in `corpus_stats.py` mines `sarl`, `sas`, `eurl`, `sci`, `ei` directly from test/train corpus frequencies.

---

## 4. Test Verification Results
All 18 unit tests passed via `pytest`:
- `tests/test_preprocessing.py`: 11 passed
- `tests/test_features.py`: 7 passed

---

## 5. Handoff to Person 3 & Person 4
- **Person 3:** Ingest `build_features(candidates_df, records_df, retrieval_events_df)` to generate `features.parquet` for training folds and OOF evaluations.
- **Person 4:** Preprocessing (`preprocess_records_df`) and feature engineering (`build_features`) are pure, stateless, and ready for end-to-end integration in `src/inference/pipeline.py`.
