"""
Sequence Data Preparation — builds temporal sequences from the feature matrix.

Each driver's race history is a chronological sequence. For each prediction,
we use the last N races as context for the LSTM/Transformer.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Sequence configuration
DEFAULT_SEQ_LENGTH = 20  # last 20 races as context
MIN_SEQ_LENGTH = 5       # minimum races to form a valid sequence


def build_driver_sequences(
    feature_matrix: pd.DataFrame,
    seq_length: int = DEFAULT_SEQ_LENGTH,
    target_col: str = "position",
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build temporal sequences from feature matrix for sequence model training.

    For each (driver, race) pair, creates a sequence of the driver's
    last `seq_length` races as input, with the current race result as target.

    Args:
        feature_matrix: Full feature matrix with season, round, driver_id, position
        seq_length: Number of past races to include in each sequence
        target_col: Column to predict
        feature_cols: Columns to use as features (if None, auto-detect numeric)

    Returns:
        Tuple of:
            sequences: (n_samples, seq_length, n_features) padded sequences
            targets: (n_samples,) target values
            driver_ids: (n_samples,) driver identifier for each sample
            seq_lengths: (n_samples,) actual sequence length (before padding)
    """
    fm = feature_matrix.sort_values(["season", "round"]).copy()

    # Auto-detect feature columns
    if feature_cols is None:
        exclude = {"season", "round", "driver_id", "constructor_id", "circuit_id",
                    "position", "grid", "dnf", "is_wet", "driver_name", "constructor_name"}
        feature_cols = [c for c in fm.columns if c not in exclude and fm[c].dtype in ("float64", "float32", "int64", "int32")]

    n_features = len(feature_cols)

    all_sequences = []
    all_targets = []
    all_driver_ids = []
    all_seq_lengths = []

    # Group by driver
    for driver_id, driver_data in fm.groupby("driver_id"):
        driver_data = driver_data.sort_values(["season", "round"])

        if len(driver_data) < MIN_SEQ_LENGTH + 1:
            continue

        features = driver_data[feature_cols].values
        targets = driver_data[target_col].values

        # Fill NaN with 0 for sequence features
        features = np.nan_to_num(features, nan=0.0)

        # Create sequences using sliding window
        for i in range(MIN_SEQ_LENGTH, len(driver_data)):
            # Skip if target is NaN
            if np.isnan(targets[i]):
                continue

            # Get sequence (last seq_length races before this one)
            start = max(0, i - seq_length)
            seq = features[start:i]
            actual_len = len(seq)

            # Pad if shorter than seq_length
            if actual_len < seq_length:
                padding = np.zeros((seq_length - actual_len, n_features))
                seq = np.vstack([padding, seq])

            all_sequences.append(seq)
            all_targets.append(targets[i])
            all_driver_ids.append(driver_id)
            all_seq_lengths.append(actual_len)

    if not all_sequences:
        return (
            np.zeros((0, seq_length, n_features)),
            np.zeros(0),
            np.array([]),
            np.zeros(0, dtype=int),
        )

    sequences = np.stack(all_sequences).astype(np.float32)
    targets = np.array(all_targets, dtype=np.float32)
    driver_ids = np.array(all_driver_ids)
    seq_lengths = np.array(all_seq_lengths, dtype=int)

    logger.info(
        "Built %d sequences: shape=%s, %d unique drivers, %d features",
        len(sequences), sequences.shape, len(set(all_driver_ids)), n_features,
    )

    return sequences, targets, driver_ids, seq_lengths


def build_entity_vocabularies(
    feature_matrix: pd.DataFrame,
) -> Dict[str, Dict[str, int]]:
    """
    Build integer mappings for entity embeddings.

    Returns:
        Dict with 'driver', 'constructor', 'circuit' vocabularies,
        each mapping string ID -> integer index.
    """
    vocabs = {}

    for col, name in [("driver_id", "driver"), ("constructor_id", "constructor"), ("circuit_id", "circuit")]:
        if col in feature_matrix.columns:
            unique = sorted(feature_matrix[col].dropna().unique())
            vocabs[name] = {v: i + 1 for i, v in enumerate(unique)}  # 0 reserved for unknown
            vocabs[name]["<UNK>"] = 0

    return vocabs


def get_entity_indices(
    feature_matrix: pd.DataFrame,
    vocabs: Dict[str, Dict[str, int]],
) -> Dict[str, np.ndarray]:
    """
    Convert entity columns to integer indices using vocabularies.

    Args:
        feature_matrix: DataFrame with entity columns
        vocabs: Vocabularies from build_entity_vocabularies

    Returns:
        Dict mapping entity name to integer index arrays
    """
    indices = {}

    for col, name in [("driver_id", "driver"), ("constructor_id", "constructor"), ("circuit_id", "circuit")]:
        if col in feature_matrix.columns and name in vocabs:
            vocab = vocabs[name]
            indices[name] = feature_matrix[col].map(lambda x: vocab.get(x, 0)).values

    return indices
