# GPS Input-Output Model Sensitivity

Author: Saisha Doma

This repository reproduces estimates of economic losses from disruptions to GPS services using the Leontief, Ghosh, and Inoperability Input-Output models. The implementation converts the original notebook calculations into validated Python modules, uses one matched set of dependency draws across all models, and records the configuration of each run.

## Reproduce the analysis

Install [uv](https://docs.astral.sh/uv/), clone the repository, and run:

```bash
uv sync --extra test --frozen
uv run reproduce-gps-io \
  --use-table data/source/use_tables.csv \
  --processed-dir data/processed/13sector \
  --output-dir results/reproduction \
  --figure-dir figures \
  --source-unit million_usd \
  --price-year 2023 \
  --table-vintage 2023 \
  --draws 10000 \
  --seed 42
```

The command rebuilds the canonical 13 sector tables, numerical summaries, Monte Carlo draws, figures, and a machine readable run manifest. Generated outputs are written to `results/` and `figures/`. They are ignored by Git because they can be rebuilt from the tracked code and data.

The central 95% draw interval is the interval from the 2.5th to the 97.5th percentile across 10,000 Monte Carlo draws. It describes variation under the declared dependency distributions. It is not an observational confidence interval.

## Test the implementation

```bash
uv run pytest
```

The tests cover input validation, model identities, sector alignment, deterministic simulation, statistical summaries, and figure dimensions.

## Repository contents

- `gps_io/`: validated data, model, simulation, summary, and plotting modules
- `scripts/reproduce.py`: command line reproduction pipeline
- `tests/`: unit, regression, and scientific tests
- `data/source/`: compact source table used by the analysis
- `data/processed/`: canonical 13 sector inputs rebuilt by the pipeline
- `data/manifests/`: source provenance and checksum metadata

The source data snapshot is small enough to distribute with the code. Its origin, interpretation, and SHA-256 checksum are recorded in `data/manifests/bea-use-table.json`.

## Citation

Please cite the related work:

```bibtex
@misc{oughton_major_2026,
  title = {Major {Space} {Weather} {Risks} {Identified} via {Coupled} {Physics}-{Engineering}-{Economic} {Modeling}},
  url = {https://arxiv.org/abs/2412.18032},
  doi = {10.48550/arXiv.2412.18032},
  publisher = {arXiv},
  author = {Oughton, Edward J. and Bor, Dennies K. and Weigel, Robert and Gaunt, C. Trevor and Dogan, Ridvan and Huang, Liling and Love, Jeffrey J. and Wiltberger, Michael},
  month = jun,
  year = {2026},
}
```
