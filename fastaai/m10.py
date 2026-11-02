"""Murphy10 word encoding and model-derived FastAAI statistics."""

from typing import Tuple

import numpy as np

GROUPS = ("A", "C", "DENQ", "FWY", "G", "H", "ILMV", "KR", "P", "ST")
LOOKUP = np.full(256, -1, dtype=np.int16)
for group, letters in enumerate(GROUPS):
    LOOKUP[np.frombuffer(letters.encode("ascii"), dtype=np.uint8)] = group
POWERS = np.array([10000, 1000, 100, 10, 1], dtype=np.int32)


def encode(sequence: str) -> Tuple[np.ndarray, int, np.ndarray]:
    digits = LOOKUP[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]
    counts = np.bincount(digits[digits >= 0], minlength=10).astype(np.int32)
    if len(digits) < 5:
        return np.empty(0, dtype=np.int32), 0, counts
    windows = np.lib.stride_tricks.sliding_window_view(digits, 5)
    windows = windows[np.all(windows >= 0, axis=1)]
    words = np.unique(windows @ POWERS).astype(np.int32)
    return words, len(windows), counts


def hmm_path() -> str:
    import importlib.util
    from pathlib import Path

    package = importlib.util.find_spec("fastaai")
    return str(
        Path(package.origin).parent / "00.Libraries/01.SCG_HMMs/Complete_SCG_DB.hmm"
    )


def create_stats_table(cursor) -> None:
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS genome_acc_stats "
        "(genome INTEGER, accession INTEGER, windows INTEGER, "
        "group_counts BLOB, protein TEXT, PRIMARY KEY (genome, accession))"
    )
