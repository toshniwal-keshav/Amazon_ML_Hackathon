import csv
from pathlib import Path

import pandas as pd


def load_source_tsv(file_path: str | Path) -> pd.DataFrame:
    """
    Load a challenge TSV safely.

    - Explicit tab delimiter
    - Keep every column as string
    - Preserve empty fields as empty strings
    - Treat quotes as literal characters
    - Validate entity_id existence and uniqueness
    """
    df = pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=[],
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
    )

    assert "entity_id" in df.columns, f"Missing entity_id in {file_path}"
    assert df["entity_id"].is_unique, (
        f"Duplicate entity_ids detected in {file_path}"
    )

    return df


def iter_source_tsv(
    file_path: str | Path,
    chunksize: int = 50_000,
):
    """
    Read a challenge TSV in chunks.

    Every chunk follows the same type and missing-value rules
    as load_source_tsv().
    """
    return pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=[],
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
        chunksize=chunksize,
    )