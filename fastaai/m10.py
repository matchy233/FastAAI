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


def frequencies(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=-1, keepdims=True)
    return np.divide(
        counts, totals, out=np.zeros_like(counts, dtype=float), where=totals > 0
    )


def load_stats(connection, genomes: int, schema: str = "main"):
    windows = np.zeros((122, genomes), dtype=float)
    counts = np.zeros((122, genomes, 10), dtype=float)
    for genome, accession, size, composition in connection.execute(
        f"SELECT genome,accession,windows,group_counts FROM {schema}.genome_acc_stats"
    ):
        windows[accession, genome] = size
        counts[accession, genome] = np.frombuffer(composition, dtype=np.int32)
    return windows, frequencies(counts)


def file_stats(protein_stats, accession_index):
    windows = np.zeros((122, 1), dtype=float)
    counts = np.zeros((122, 1, 10), dtype=float)
    for accession, (size, composition, _protein) in protein_stats.items():
        index = accession_index[accession]
        windows[index, 0] = size
        counts[index, 0] = composition
    return windows, frequencies(counts)


def pool_statistics(intersections, q_windows, q_frequencies, t_windows, t_frequencies):
    minimum = np.minimum(q_windows[:, None], t_windows)
    shared = minimum > 0
    r = np.einsum("ag,atg->at", q_frequencies, t_frequencies)
    chance = (q_windows[:, None] * t_windows - minimum) * r**5
    s = np.sum(np.where(shared, intersections, 0), axis=0)
    lam = np.sum(np.where(shared, chance, 0), axis=0)
    m = np.sum(minimum, axis=0)
    ratio = np.divide(
        s - lam, m, out=np.full(len(m), np.nan), where=(m > 0) & (s > lam)
    )
    return {
        "P_hat": ratio**0.2,
        "shared_SCPs": shared.sum(axis=0),
        "S": s,
        "lambda": lam,
        "m": m,
    }


def query_statistics(intersections, accessions, query, query_stats, target_stats):
    return pool_statistics(
        intersections,
        query_stats[0][accessions, query],
        query_stats[1][accessions, query, :],
        target_stats[0][accessions, :],
        target_stats[1][accessions, :, :],
    )
