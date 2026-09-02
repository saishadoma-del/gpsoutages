# Role: Reproduction entry point for the GPS input-output analysis
# Author: Saisha Doma
# Description: Builds the 13 sector inputs, runs matched Monte Carlo simulations,
# writes numerical summaries and figures, and records a provenance manifest.

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
from datetime import datetime
from datetime import timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import matplotlib
import numpy as np
import pandas as pd

from gps_io.constants import GPS_DEPENDENCY_RANGES
from gps_io.constants import OUTAGE_HOURS
from gps_io.constants import SECTOR_GROUPS
from gps_io.data import build_technology_tables_from_csv
from gps_io.data import write_technology_tables
from gps_io.models import InputOutputModel
from gps_io.plotting import BAR_HEIGHT_AXIS_UNITS
from gps_io.plotting import FIGURE_DIMENSIONS_INCHES
from gps_io.plotting import FIGURE_DPI
from gps_io.plotting import PAPER_FIGURE_NAMES
from gps_io.plotting import PAPER_SELECTED_FIGURE_NAMES
from gps_io.plotting import plot_duration_panels
from gps_io.simulation import draw_dependency_fractions
from gps_io.simulation import scale_to_outage
from gps_io.simulation import simulate_all_models
from gps_io.summaries import QUANTILES
from gps_io.summaries import summarize_economy
from gps_io.summaries import summarize_sectors

log = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_dataframe_csv(
    frame: pd.DataFrame,
    path: Path,
    *,
    index: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        frame.to_csv(temporary_path, index=index)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_json(data: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as temporary:
            json.dump(data, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _summary_frames(
    simulations: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sector_frames: list[pd.DataFrame] = []
    economy_frames: list[pd.DataFrame] = []

    for model_name, annual_simulation in simulations.items():
        for hours in OUTAGE_HOURS:
            simulation = scale_to_outage(annual_simulation, hours)
            for effect, summary in summarize_sectors(simulation).items():
                frame = summary.rename_axis("sector").reset_index()
                frame.insert(0, "effect", effect)
                frame.insert(0, "duration_hours", hours)
                frame.insert(0, "model", model_name)
                sector_frames.append(frame)

            economy = summarize_economy(simulation).rename_axis("effect").reset_index()
            economy.insert(0, "duration_hours", hours)
            economy.insert(0, "model", model_name)
            economy_frames.append(economy)

    return (
        pd.concat(sector_frames, ignore_index=True),
        pd.concat(economy_frames, ignore_index=True),
    )


def reproduce(args: argparse.Namespace) -> None:
    use_table_path = args.use_table.resolve()
    output_dir = args.output_dir.resolve()
    figure_dir = args.figure_dir.resolve()
    processed_dir = args.processed_dir.resolve()

    log.info("Building technology tables from %s", use_table_path)
    tables = build_technology_tables_from_csv(
        use_table_path,
        source_unit=args.source_unit,
        price_year=args.price_year,
        table_vintage=args.table_vintage,
    )
    if not args.figures_only:
        write_technology_tables(
            tables,
            processed_dir,
            year_label=str(args.price_year),
        )

    model = InputOutputModel.from_tables(tables)
    dependency_draws = draw_dependency_fractions(
        model.sectors,
        GPS_DEPENDENCY_RANGES,
        draws=args.draws,
        seed=args.seed,
    )
    simulations = simulate_all_models(model, dependency_draws)

    if not args.figures_only:
        sector_summary, economy_summary = _summary_frames(simulations)
        _atomic_dataframe_csv(
            dependency_draws,
            output_dir / "dependency_draws.csv",
            index=True,
        )
        _atomic_dataframe_csv(
            sector_summary,
            output_dir / "sector_summary.csv",
            index=False,
        )
        _atomic_dataframe_csv(
            economy_summary,
            output_dir / "economy_summary.csv",
            index=False,
        )

    for model_name, simulation in simulations.items():
        figure_path = figure_dir / PAPER_FIGURE_NAMES[model_name]
        plot_duration_panels(simulation, figure_path, price_year=args.price_year)
        plot_duration_panels(
            simulation,
            figure_path.with_suffix(".pdf"),
            price_year=args.price_year,
        )
        if model_name in {"ghosh", "iim"}:
            selected_path = (
                figure_dir / PAPER_SELECTED_FIGURE_NAMES[model_name]
            )
            plot_duration_panels(
                simulation,
                selected_path,
                price_year=args.price_year,
                selected_sectors=("MANUF", "TRANS", "AGR"),
            )
            plot_duration_panels(
                simulation,
                selected_path.with_suffix(".pdf"),
                price_year=args.price_year,
                selected_sectors=("MANUF", "TRANS", "AGR"),
            )

    log.info("Wrote figures to %s", figure_dir)
    if args.figures_only:
        return

    manifest: dict[str, object] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": args.use_table.as_posix(),
            "sha256": _sha256(use_table_path),
            "source_unit": args.source_unit,
            "table_vintage": args.table_vintage,
            "price_year": args.price_year,
        },
        "model": {
            "sector_groups": {
                sector: list(codes)
                for sector, codes in SECTOR_GROUPS.items()
            },
            "dependency_ranges": {
                sector: list(bounds)
                for sector, bounds in GPS_DEPENDENCY_RANGES.items()
            },
            "models": list(simulations),
            "outage_hours": list(OUTAGE_HOURS),
            "duration_assumption": "Annual flows are uniform across 8760 hours.",
            "loss_sign": "Positive values are loss magnitudes.",
        },
        "simulation": {
            "draws": args.draws,
            "seed": args.seed,
            "distribution": "independent uniform by sector",
            "matched_draws_across_models": True,
            "uncertainty_interval": {
                "definition": "central interval across Monte Carlo draws",
                "quantiles": list(QUANTILES),
            },
        },
        "figures": {
            "dimensions_inches": {
                name: list(dimensions)
                for name, dimensions in FIGURE_DIMENSIONS_INCHES.items()
            },
            "dpi": FIGURE_DPI,
            "bar_height_axis_units": BAR_HEIGHT_AXIS_UNITS,
            "bbox_inches": None,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    _atomic_json(manifest, output_dir / "manifest.json")
    log.info("Wrote reproduction outputs to %s", output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the GPS input-output model sensitivity analysis."
    )
    parser.add_argument(
        "--use-table",
        type=Path,
        required=True,
        help="Path to the compact BEA Use Table CSV.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/13sector"),
        help="Directory for canonical 13 sector technology tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/corrected"),
        help="Directory for summaries, draws, and the run manifest.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("figures"),
        help="Directory containing only rendered PNG and PDF figures.",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Render figures without writing tables, summaries, draws, or a manifest.",
    )
    parser.add_argument(
        "--source-unit",
        choices=("million_usd", "billion_usd"),
        default="million_usd",
    )
    parser.add_argument("--price-year", type=int, default=2023)
    parser.add_argument("--table-vintage", type=int, default=2023)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    reproduce(build_parser().parse_args())


if __name__ == "__main__":
    main()
