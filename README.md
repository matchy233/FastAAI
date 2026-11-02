# FastAAI-M10

A research fork of FastAAI. Branch `m10-master` is upstream master 3b4904a
(including upstream PRs #27-#30) plus the M10 changes. It retains the native
122-marker HMM search, best-hit rules, SQLite inverted index, and
multiprocessing, including upstream's task-based query workers. It replaces
amino-acid tetramers and the native conversion with Murphy10 5-mers and an
LG-model inverse. The upstream paper states the MIT license.

Branch `development` carries the same M10 changes on the FastAAI 0.1.20 release
file; the William project's `tools/fastaai_m10` package is a copy of that branch.
Both branches give byte-identical database contents and query outputs on the
validation panels.

## Usage

With the checkout on `PYTHONPATH` (the William project's Python 3.10
environment is `.venv-fastaai`):

```sh
export PYTHONPATH=/path/to/FastAAI
python -m fastaai build_db --proteins /path/to/proteomes --output build --threads 4
python -m fastaai db_query --query build/database/FastAAI_database.sqlite.db --target build/database/FastAAI_database.sqlite.db --output query --threads 4 --in_memory
python -m fastaai simple_query --proteins /path/to/queries --target build/database/FastAAI_database.sqlite.db --output simple --threads 4 --create_query_db
python -m fastaai single_query --query_protein query.faa --target_protein target.faa --output single --threads 2
python -m fastaai merge_db --recipient recipient.db --donors donor.db --threads 4
```

The `fastaai/fastaai` wrapper runs the same commands without `PYTHONPATH`; it
adds its own checkout to `sys.path` only when that path is not already there.
Validation used NumPy 1.26.4, pyhmmer 0.7.4 and pyrodigal 1.0.2 on protein
inputs; upstream's genome-input path uses `pyrodigal.GeneFinder` (pyrodigal 2 or
later; `setup.py` pins pyrodigal 3.6.3). The numerical query code adds no
dependency. Use databases built by this fork: native tetramer databases do not
have its 5-mer words or per-marker statistics. `merge_db` combines M10 databases.

Add `--output_style matrix` to `db_query` or `simple_query` for both
`results/FastAAI_matrix.txt` (AAI) and `results/FastAAI_t_hat_matrix.txt` (distance).
`db_query --store_results` and in-memory/on-disk query paths are supported,
including more threads than query genomes.
`--do_stdev` is retained for CLI compatibility; the M10 interval uses the fixed
calibration residuals, not the native per-marker Jaccard SD.

## Statistic

Murphy10 groups are `A C DENQ FWY G H ILMV KR P ST` (Bonsai commit
534a5f8cea58ccad2d084d124b3e50f298d1ce78). Valid 5-mers receive a base-10 int32
code; each protein stores a set of unique words. A window containing a
nonstandard residue is skipped. `genome_acc_stats` stores valid window count,
10 group counts, and the selected protein ID for each genome/marker.

For each shared marker, S is the word-set intersection size, m is the smaller
valid window count, r is the inner product of group-frequency vectors, and
lambda = (m_x m_y - m) r^5. Pool across markers:

```
P_hat = ((sum(S) - sum(lambda)) / sum(m))^(1/5)
```

A nonpositive excess or zero denominator yields a missing estimate. The fixed
LG transition model maps `P_hat` to expected substitutions per site, then to
expected original-alphabet marker identity. This assumes the pooled statistic
can be described by the model; it does not calculate an alignment.

## Output columns

| Column | Meaning |
| --- | --- |
| query, target | Input filename identifiers, consistent across supported commands |
| shared_SCPs | Shared markers with at least one valid 5-mer window on both sides |
| P_hat | Background-corrected pooled Murphy10 match fraction |
| t_hat | LG expected substitutions/site; model-derived, not a fitted AAI |
| marker_identity | LG-implied AA20 identity, in percent |
| AAI | Quadratic fitted estimate of DIAMOND-RBH whole-proteome mean identity, in percent |
| AAI_low, AAI_high | Empirical 95% residual interval, clipped to [0,100] |
| AAI_saturated | `t_hat > 0.8`; missing when distance is missing |

Missing numerical fields are `N/A`. Self rows, identified by matching filenames,
are explicitly reported as P=1, t=0, identity=AAI=100, interval [100,100]. This is a
reporting convention: unique-word counting plus repeated words does not itself
produce an exact self estimate. Filenames identify genomes, so do not reuse the
same filename for different genomes across query and target databases.
File queries (`simple_query`, `single_query`) add markers in a different order
than database queries, so their values can differ in the last digits (at most
1.4e-14 AAI points in the two-genome CLI check).

AAI uses `a + b I + c I^2`, with I a **fraction**, and coefficients in
`conversion.json`. It is fitted once to 480 Panel B genus/family pairs across
32 taxonomic families; reported Panel B accuracy uses leave-one-family-out fits.
Residual SD bins use those out-of-fold predictions. The shipped full fit is for
subsequent inference; its training fit is not held-out validation.

## Validation and limitations

- All 14,909 B and 7,611 A marker selections, proteins, word sets and metadata
  match the prior U sets. All 480 B and 4,560 A pairs reproduce their frozen
  Murphy10 statistics; maximum P difference is below 2e-16.
- The dense LG table is checked against exact inversion on 1,000 points over
  t=0..5 (including all observed panel distances), within 1e-6. Its provenance
  and generator hash are in `lg_tables.json`.
- Panel B LOFO AAI RMSE is 3.018/2.301 points at genus/family. Panel A bacterial
  transfer RMSE is 2.419/2.267, with empirical interval coverage 95.2%/91.3%.
  Archaea transfer is poorer: bias -3.59/-4.67, RMSE 3.89/4.89, coverage 83.3%/22.2%.
- AAI intervals are empirical and have no universal coverage guarantee. Deep-rank
  AAI extrapolation is not validated: the quadratic is nonmonotonic outside its
  training range and must not be interpreted as a deep-distance scale. Use
  `t_hat` for the distance output; the saturation flag does not validate AAI.
- LG distance is a pooled marker estimate, not necessarily identical to
  alignment-ML LG distance on another gene set. Different sets, aggregation,
  evolutionary models and rate variation all affect that comparison.
- Protein prediction is reused from saved inputs for the M10 build timing.
  Compare matched native protein-input controls; historical genome-input builds
  additionally include protein prediction. Query batch timing includes startup
  and output I/O, ordered pairs and self rows; it is not an optimized-tool claim.

Run the tests from the checkout root with `python -m unittest discover -s tests`.
`tests/test_m10.py` checks encoding, pooling, the LG inverse and the conversion
against reference values frozen in `tests/data/m10_reference.json.gz` (with the
hashes of the reference sources). The upstream query tests build minimal M10
databases. Run records in the William project:
`runs/estimator/fastaai-m10/2026-11-02/SUMMARY.md` (development) and
`runs/estimator/fastaai-m10-rebase/2026-11-02/SUMMARY.md` (this branch).
