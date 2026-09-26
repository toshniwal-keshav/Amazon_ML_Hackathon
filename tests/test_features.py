"""
Unit tests for feature engineering (Person 2).
Verifies mathematical correctness, edge cases, and real evidence examples from training_match_examples.md.
"""

import pytest
import numpy as np
import pandas as pd

from src.preprocessing.normalize import preprocess_records_df, normalize_aggressive
from src.features.name_features import compute_name_feature_dict
from src.features.address_features import compute_address_feature_dict
from src.features.cross_features import compute_cross_feature_dict
from src.features.retrieval_features import pivot_retrieval_features
from src.features.build import build_features, FEATURE_COLUMNS


def test_example_66_kalyani_containment():
    """
    The Kalyani Case (Example 66):
    'Kalyani Welfare Society' matches 'Kalyani'.
    Symmetric similarities are moderate (~0.4), but containment of candidate in S1 must be 1.0!
    """
    s1_name = "kalyani welfare society"
    cand_name = "kalyani"

    feat = compute_name_feature_dict(s1_name, cand_name)
    assert feat["name_containment_cand_in_s1"] >= 0.99
    assert feat["name_containment_s1_in_cand"] < 0.5
    assert feat["name_first_token_match"] == 1.0


def test_example_69_nexgild_address_match():
    """
    The NEXGILD Case (Example 69):
    'Oncology Associates' matches 'NEXGILD' purely on address.
    Address features must show strong similarity while name features are near-zero.
    """
    s1_name = "oncology associates"
    cand_name = "nexgild"
    s1_addr = "2020 santa monica blvd suite 600 santa monica ca"
    cand_addr = "2020 santa monica blvd suite 600 santa monica ca"

    name_feat = compute_name_feature_dict(s1_name, cand_name)
    addr_feat = compute_address_feature_dict(s1_addr, cand_addr)

    assert name_feat["name_char_cos"] < 0.20
    assert addr_feat["addr_char_cos"] > 0.95
    assert addr_feat["addr_exact_cons"] == 1.0
    assert addr_feat["addr_first_num_match"] == 1.0


def test_example_71_blank_address_nan_handling():
    """
    The Blank Address Case (Example 71):
    Candidate has an empty address string and is a true match.
    Must produce addr_missing_cand=1, and NaN for similarity metrics (never impute 0.0!).
    """
    s1_addr = "123 Main St, Springfield, IL"
    cand_addr = ""

    addr_feat = compute_address_feature_dict(s1_addr, cand_addr)
    cross_feat = compute_cross_feature_dict(
        s1_name_raw="Test S1",
        cand_name_raw="Test S2",
        s1_addr_raw=s1_addr,
        cand_addr_raw=cand_addr,
        s1_country_norm="US",
        cand_country_norm="US",
        name_char_cos=0.85,
        addr_char_cos=addr_feat["addr_char_cos"],
    )

    assert cross_feat["addr_missing_cand"] == 1.0
    assert cross_feat["addr_missing_either"] == 1.0
    assert np.isnan(addr_feat["addr_char_cos"])
    assert np.isnan(addr_feat["addr_num_jaccard"])
    assert np.isnan(addr_feat["addr_token_sort_ratio"])
    assert cross_feat["country_eq"] == 1.0


def test_example_71_digit_typo_soft_conflict():
    """
    The Digit Typo Case (Example 71):
    '703 Beacon Court' vs '70 Beacon Court' (dropped digit).
    Must produce addr_num_conflict=1.0 and addr_num_jaccard=0.0 without throwing errors.
    """
    s1_addr = "703 beacon court"
    cand_addr = "70 beacon court"

    addr_feat = compute_address_feature_dict(s1_addr, cand_addr)
    assert addr_feat["addr_num_conflict"] == 1.0
    assert addr_feat["addr_num_jaccard"] == 0.0
    assert addr_feat["addr_first_num_match"] == 0.0
    assert addr_feat["addr_char_cos"] > 0.70  # Street text is still very similar!


def test_example_68_damani_mixed_script():
    """
    The Damani Mixed-Script Case (Example 68):
    Records contain mixed Latin and Devanagari script ('दिल्ली' vs 'Delhi').
    """
    s1_name = "damani technologies llp"
    cand_name = "damani technologies l l p"
    s1_addr = "plot no u 1 22a khasra no 83 22 1st floor phase 1 budh vihar north west delhi"
    cand_addr = "plot no u 1 22a khasra no 83 22 1st floor phase 1 budh vihar दिल्ली"

    s1_aggr = normalize_aggressive(s1_name)
    cand_aggr = normalize_aggressive(cand_name)

    name_feat = compute_name_feature_dict(s1_name, cand_name, s1_aggr, cand_aggr)
    addr_feat = compute_address_feature_dict(s1_addr, cand_addr)

    assert name_feat["name_exact_aggr"] == 1.0 or name_feat["name_char_cos"] > 0.80
    assert addr_feat["addr_char_cos"] > 0.75  # Shared plot numbers and street prefix
    assert addr_feat["addr_num_conflict"] == 0.0  # Common digits matched


def test_pivot_retrieval_features():
    retrieval_events = pd.DataFrame([
        {"pair_key": "S1-1::S2-2", "route": "exact_name", "rank": 1, "score": 1.0},
        {"pair_key": "S1-1::S2-2", "route": "tfidf_name", "rank": 3, "score": 0.92},
        {"pair_key": "S1-1::S3-3", "route": "numeric", "rank": 5, "score": 0.75},
    ])
    pivoted = pivot_retrieval_features(retrieval_events, pair_keys=["S1-1::S2-2", "S1-1::S3-3", "S1-1::S2-99"])
    assert len(pivoted) == 3
    assert "n_routes" in pivoted.columns
    assert pivoted.loc[pivoted["pair_key"] == "S1-1::S2-2", "n_routes"].values[0] == 2.0
    assert pivoted.loc[pivoted["pair_key"] == "S1-1::S2-2", "retrieved_by_exact_name"].values[0] == 1.0
    assert pivoted.loc[pivoted["pair_key"] == "S1-1::S2-99", "retrieved_by_exact_name"].values[0] == 0.0


def test_build_features_pipeline():
    # Construct mock records
    records_raw = pd.DataFrame([
        {"entity_id": "S1-100", "name": "Kalyani Welfare Society", "address": "Kadugodi, Whitefield, Bangalore", "country": "India"},
        {"entity_id": "S2-200", "name": "Kalyani", "address": "Whitefield, Bangalore, Karnataka", "country": "India"},
        {"entity_id": "S3-300", "name": "Oncology Associates", "address": "2020 Santa Monica Blvd", "country": "US"},
        {"entity_id": "S2-400", "name": "NEXGILD", "address": "2020 Santa Monica Blvd", "country": "US"},
    ])
    records_df = preprocess_records_df(records_raw)

    # Candidate pairs
    candidates_df = pd.DataFrame([
        {"s1_id": "S1-100", "candidate_id": "S2-200"},
        {"s1_id": "S3-300", "candidate_id": "S2-400"},
    ])

    retrieval_events = pd.DataFrame([
        {"pair_key": "S1-100::S2-200", "route": "exact_name", "rank": 1, "score": 1.0},
        {"pair_key": "S3-300::S2-400", "route": "tfidf_address", "rank": 1, "score": 0.98},
    ])

    features_df = build_features(candidates_df, records_df, retrieval_events)

    # Invariant assertions
    assert len(features_df) == 2
    assert "pair_key" in features_df.columns
    assert "s1_id" in features_df.columns
    assert "candidate_id" in features_df.columns
    assert features_df.loc[0, "pair_key"] == "S1-100::S2-200"
    assert features_df.loc[1, "pair_key"] == "S3-300::S2-400"

    # All required feature columns exist
    for col in FEATURE_COLUMNS:
        assert col in features_df.columns, f"Missing feature column: {col}"

    # Pair 0 (Kalyani) checks
    assert features_df.loc[0, "name_containment_cand_in_s1"] >= 0.99
    assert features_df.loc[0, "country_eq"] == 1.0

    # Pair 1 (NEXGILD) checks
    assert features_df.loc[1, "addr_char_cos"] > 0.90
    assert features_df.loc[1, "name_char_cos"] < 0.20
