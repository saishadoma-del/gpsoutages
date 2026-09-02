# Role: Statistical summaries for input-output loss simulations
# Author: Saisha Doma
# Description: Computes sector and economy statistics with aggregate intervals
# calculated from draw totals rather than sums of sector quantiles.

from __future__ import annotations

import pandas as pd

from gps_io.simulation import LossSimulation

QUANTILES = (0.025, 0.975)


def _statistics(values: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "low_2.5pct": values.quantile(QUANTILES[0]),
            "median": values.median(),
            "mean": values.mean(),
            "high_97.5pct": values.quantile(QUANTILES[1]),
        }
    )


def summarize_sectors(
    simulation: LossSimulation,
) -> dict[str, pd.DataFrame]:
    simulation.validate()
    return {
        "direct": _statistics(simulation.direct),
        "indirect": _statistics(simulation.indirect),
        "total": _statistics(simulation.total),
    }


def summarize_economy(
    simulation: LossSimulation,
    *,
    sectors: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    simulation.validate()
    sector_ids = (
        list(simulation.total.columns)
        if sectors is None
        else list(sectors)
    )
    missing = sorted(set(sector_ids) - set(simulation.total.columns))
    if missing:
        raise ValueError(f"Summary sectors do not exist: {missing}")
    if len(sector_ids) != len(set(sector_ids)):
        raise ValueError("Summary sectors must be unique")
    effects = {
        "direct": simulation.direct[sector_ids].sum(axis=1),
        "indirect": simulation.indirect[sector_ids].sum(axis=1),
        "total": simulation.total[sector_ids].sum(axis=1),
    }
    records = {
        effect: {
            "low_2.5pct": values.quantile(QUANTILES[0]),
            "median": values.median(),
            "mean": values.mean(),
            "high_97.5pct": values.quantile(QUANTILES[1]),
        }
        for effect, values in effects.items()
    }
    return pd.DataFrame.from_dict(records, orient="index")
