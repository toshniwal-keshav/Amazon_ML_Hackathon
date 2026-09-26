"""
Unit tests for preprocessing and normalization (Person 2).
Covers multilingual safety, accent normalization, Devanagari preservation, and corpus mining.
"""

import os
import json
import pytest
import pandas as pd
import numpy as np

from src.preprocessing.normalize import (
    normalize_conservative,
    normalize_aggressive,
    normalize_country,
    preprocess_record,
    preprocess_records_df,
)
from src.preprocessing.corpus_stats import (
    mine_corpus_suffixes,
    mine_dba_markers,
    save_mined_vocab,
    load_mined_vocab,
)


def test_conservative_normalization_basic():
    text = "  Technology  BBM  Consultants Private Limited.  "
    norm = normalize_conservative(text)
    assert norm == "technology bbm consultants private limited"


def test_conservative_normalization_null_and_empty():
    assert normalize_conservative(None) == ""
    assert normalize_conservative("") == ""
    assert normalize_conservative(np.nan) == ""
    assert normalize_conservative("   ") == ""


def test_conservative_normalization_punctuation():
    text = "Allied Trusted Jewelers (LLC) - [Unit #3204] / 1088-E3"
    norm = normalize_conservative(text)
    assert norm == "allied trusted jewelers llc unit 3204 1088 e3"


def test_multilingual_french_accents():
    # French accented characters should decompose combining diacritics into base letters
    text = "Société Générale de Crédit Français — Établissement Privé"
    norm = normalize_conservative(text)
    assert "societe" in norm
    assert "generale" in norm
    assert "credit" in norm
    assert "francais" in norm
    assert "etablissement" in norm


def test_multilingual_devanagari_no_corruption():
    # Devanagari characters and vowel signs/matras must NOT be stripped or corrupted
    text = "दामिनी टेक्नोलॉजीज LLP — Phase-1, Budh Vihar, दिल्ली"
    norm = normalize_conservative(text)
    # Assert Hindi words stay intact
    assert "दामिनी" in norm
    assert "टेक्नोलॉजीज" in norm
    assert "दिल्ली" in norm
    assert "llp" in norm
    assert "phase 1" in norm


def test_aggressive_normalization_english_and_indian_suffixes():
    text = "Kalyani Welfare Society Private Limited"
    aggr = normalize_aggressive(text)
    assert aggr == "kalyani welfare"

    text2 = "Damani Technologies L.L.P."
    aggr2 = normalize_aggressive(text2)
    assert aggr2 == "damani technologies"


def test_aggressive_normalization_french_suffixes():
    # French legal suffixes (SARL, SAS, EURL, SCI)
    text1 = "Dupond Transport SARL"
    assert normalize_aggressive(text1) == "dupond transport"

    text2 = "Atelier de Coiffure S.A.S."
    assert normalize_aggressive(text2) == "atelier de coiffure"

    text3 = "Immobiliere Saint Germain SCI"
    assert normalize_aggressive(text3) == "immobiliere saint germain"


def test_aggressive_normalization_leading_markers():
    text1 = "M/s Shirdi Corp Services"
    aggr1 = normalize_aggressive(text1)
    assert aggr1 == "shirdi" or "shirdi corp" in aggr1

    text2 = "*** Lotus Care"
    assert normalize_aggressive(text2) == "lotus care"

    text3 = "C/o Suman Prajapati"
    assert normalize_aggressive(text3) == "suman prajapati"


def test_country_normalization():
    assert normalize_country("us") == "US"
    assert normalize_country("  India  ") == "INDIA"
    assert normalize_country("FRANCE") == "FRANCE"
    assert normalize_country(None) == ""
    assert normalize_country(np.nan) == ""


def test_corpus_stats_mining(tmp_path):
    sample_names = [
        "Apex Logistics LLC",
        "Beta Systems LLC",
        "Delta Solutions Inc",
        "Gamma Tech Inc",
        "Epsilon Corp SARL",
        "Zeta Consulting SARL",
        "Kalyani Welfare Society Private Limited",
        "Damani Technologies Private Limited",
    ]
    mined = mine_corpus_suffixes(sample_names, min_freq_ratio=0.01)
    assert "llc" in mined
    assert "inc" in mined
    assert "sarl" in mined
    assert "private limited" in mined

    # Test save and load
    vocab_path = tmp_path / "suffix_tokens.json"
    save_mined_vocab(mined, filepath=vocab_path)
    loaded = load_mined_vocab(filepath=vocab_path)
    assert "sarl" in loaded
    assert "llc" in loaded


def test_preprocess_records_df():
    data = pd.DataFrame([
        {
            "entity_id": "S1-101",
            "name": "Acme Corp LLC",
            "address": "123 Main St, New York, NY",
            "country": "US",
        },
        {
            "entity_id": "S2-202",
            "name": "Acme Corp",
            "address": "123 Main Street, NY",
            "country": "US",
        },
    ])
    records_df = preprocess_records_df(data)
    expected_cols = [
        "entity_id", "source", "raw_name", "raw_address", "raw_country",
        "norm_name_cons", "norm_name_aggr", "norm_address_cons", "country_norm"
    ]
    for col in expected_cols:
        assert col in records_df.columns

    assert records_df.loc[0, "norm_name_cons"] == "acme corp llc"
    assert records_df.loc[0, "norm_name_aggr"] == "acme"
    assert records_df.loc[0, "source"] == "S1"
    assert records_df.loc[1, "source"] == "S2"
