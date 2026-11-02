"""Tests for the Murphy10 statistics against frozen reference values.

tests/data/m10_reference.json.gz holds values computed by the reference
implementation (kmer_identity in the William project) with the sources' hashes
under "provenance"; the cases mirror that project's tools/fastaai_m10/tests.
"""

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fastaai import m10

FIXTURE = Path(__file__).resolve().parent / "data" / "m10_reference.json.gz"
with gzip.open(FIXTURE, "rt", encoding = "ascii") as _handle:
	REFERENCE = json.load(_handle)


def _pool_from_sequences(x_sequences, y_sequences):
	"""Pool M10 statistics over paired marker sequences of one genome pair."""
	encoded = [(m10.encode(a), m10.encode(b)) for a, b in zip(x_sequences, y_sequences)]
	shared = np.array([[len(set(a[0]) & set(b[0]))] for a, b in encoded])
	mx = np.array([a[1] for a, b in encoded], dtype = float)
	my = np.array([[b[1]] for a, b in encoded], dtype = float)
	fx = m10.frequencies(np.array([a[2] for a, b in encoded], dtype = float))
	fy = m10.frequencies(np.array([[b[2]] for a, b in encoded], dtype = float))
	return m10.pool_statistics(shared, mx, fx, my, fy)


class EncodingTests(unittest.TestCase):
	"""Murphy10 5-mer words, valid window counts and group compositions."""

	def test_fifty_random_proteins_match_reference_word_sets(self):
		"""Match the reference word set, window count and composition exactly."""
		self.assertEqual(len(REFERENCE["encoding"]), 50)
		for case in REFERENCE["encoding"]:
			words, windows, counts = m10.encode(case["sequence"])
			self.assertEqual(words.dtype, np.int32)
			self.assertEqual(words.tolist(), case["words"])
			self.assertEqual(windows, case["windows"])
			np.testing.assert_allclose(counts / counts.sum(), case["composition"], rtol = 0, atol = 1e-15)

	def test_unknown_windows_and_repeats(self):
		"""Skip windows with unknown residues; count repeated words once."""
		words, windows, counts = m10.encode("AAAAAAXAAAAAA")
		self.assertEqual(words.tolist(), [0])
		self.assertEqual(windows, 4)
		self.assertEqual(counts.tolist(), [12, 0, 0, 0, 0, 0, 0, 0, 0, 0])
		self.assertEqual(m10.encode("ACD")[1], 0)


class PoolingTests(unittest.TestCase):
	"""Chance-corrected pooled match fraction."""

	def test_fifty_protein_pairs_match_reference_kmer_score(self):
		"""Match the single-protein reference P_hat and background within 1e-12."""
		self.assertEqual(len(REFERENCE["protein_pairs"]), 50)
		for case in REFERENCE["protein_pairs"]:
			result = _pool_from_sequences([case["x"]], [case["y"]])
			np.testing.assert_allclose(result["P_hat"][0], case["p_hat"], rtol = 0, atol = 1e-12)
			np.testing.assert_allclose(result["lambda"][0], case["lambda"], rtol = 0, atol = 1e-12)

	def test_five_genome_pairs_match_reference_pool(self):
		"""Match the reference genome-pair pool within 1e-12."""
		sequences = REFERENCE["genome_pairs"]["sequences"]
		pairs = REFERENCE["genome_pairs"]["pairs"]
		self.assertEqual(len(pairs), 5)
		for pair in pairs:
			x = sequences[pair["genome_x"]]
			y = sequences[pair["genome_y"]]
			common = sorted(set(x) & set(y))
			result = _pool_from_sequences([x[k] for k in common], [y[k] for k in common])
			np.testing.assert_allclose(result["P_hat"][0], pair["P"], rtol = 0, atol = 1e-12)
			self.assertEqual(len(common), pair["labels_shared"])
			self.assertEqual(int(result["shared_SCPs"][0]), pair["labels_defined"])


class LgTableTests(unittest.TestCase):
	"""Interpolated LG inverse against exact inversion."""

	def test_thousand_points_match_exact_inversion(self):
		"""Agree within 1e-6 on t = 0..5, which covers the panel distances."""
		lg = REFERENCE["lg"]
		self.assertEqual(len(lg["t"]), 1000)
		result = m10.convert(np.array(lg["P"]))
		np.testing.assert_allclose(result["t_hat"], lg["t_inverse"], rtol = 0, atol = 1e-6)
		np.testing.assert_allclose(result["marker_identity"], lg["marker_identity"], rtol = 0, atol = 1e-6)


class ConversionTests(unittest.TestCase):
	"""Output units, missing values, intervals and self pairs."""

	def test_output_units_intervals_and_missing_are_explicit(self):
		"""Report exact zero distance, ordered intervals and N/A for missing values."""
		result = m10.convert(np.array([1.0, 0.5, m10.FLOOR, np.nan]))
		self.assertEqual(result["t_hat"][0], 0)
		self.assertTrue(np.isclose(result["marker_identity"][0], 100))
		self.assertTrue(np.isnan(result["AAI"][2:]).all())
		self.assertTrue(np.all(result["AAI_low"][:2] < result["AAI"][:2]))
		self.assertTrue(np.all(result["AAI"][:2] < result["AAI_high"][:2]))
		with tempfile.TemporaryDirectory() as tempdir:
			path = Path(tempdir) / "out.tsv"
			m10.write_tsv(path, "q", ["a", "b", "c", "d"], np.array([80, 75, 45, 0]), result)
			lines = path.read_text().splitlines()
		self.assertEqual(lines[0].split("\t"), m10.HEADER)
		self.assertTrue(lines[-1].endswith("N/A\tN/A"))

	def test_intervals_stay_inside_percentage_range(self):
		"""Clip interval bounds to [0, 100] without moving the point estimate."""
		result = m10.convert(np.array([1.0, 0.99, 0.5]))
		self.assertTrue(np.all(result["AAI_high"] <= 100))
		self.assertTrue(np.all(result["AAI_low"] >= 0))
		self.assertEqual(result["AAI_high"][0], 100)
		self.assertLess(result["AAI"][0], 100)

	def test_self_pairs_are_reported_exactly_and_others_untouched(self):
		"""Set matching identifiers to the exact self values only."""
		result = m10.convert(np.array([0.99, 0.5, np.nan]))
		before = {key: np.array(value, copy = True) for key, value in result.items()}
		marked = m10.mark_self(result, "q", ["q", "other", "q"])
		for key, value in [
			("P_hat", 1),
			("t_hat", 0),
			("marker_identity", 100),
			("AAI", 100),
			("AAI_low", 100),
			("AAI_high", 100),
		]:
			self.assertEqual(marked[key][0], value)
			self.assertEqual(marked[key][2], value)
			np.testing.assert_array_equal(marked[key][1], before[key][1])
		self.assertFalse(marked["AAI_saturated"][0])
		self.assertFalse(marked["AAI_saturated"][2])
		untouched = m10.mark_self(m10.convert(np.array([0.5])), "q", ["other"])
		np.testing.assert_array_equal(untouched["AAI"], before["AAI"][1:2])


if __name__ == "__main__":
	unittest.main()
