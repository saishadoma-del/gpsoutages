# Role: Scientific tests for Monte Carlo result summaries
# Author: Saisha Doma
# Description: Ensures economy intervals are computed from totals for each draw.

import pandas as pd

from gps_io.simulation import LossSimulation
from gps_io.summaries import summarize_economy
from gps_io.summaries import summarize_sectors


def test_economy_interval_uses_total_for_each_draw() -> None:
    total = pd.DataFrame(
        [[0.0, 10.0], [10.0, 0.0], [2.0, 8.0], [8.0, 2.0]],
        columns=["A", "B"],
    )
    direct = total * 0.75
    simulation = LossSimulation(
        model="leontief",
        direct=direct,
        indirect=total - direct,
        total=total,
        monetary_unit="billion_usd",
        time_basis="per_year",
    )

    economy = summarize_economy(simulation)
    sectors = summarize_sectors(simulation)["total"]

    assert economy.loc["total", "low_2.5pct"] == 10.0
    assert economy.loc["total", "high_97.5pct"] == 10.0
    assert sectors["low_2.5pct"].sum() < 10.0

def test_economy_summary_can_select_displayed_sectors() -> None:
    total = pd.DataFrame(
        [[2.0, 10.0], [4.0, 20.0]],
        columns=["A", "B"],
    )
    direct = total * 0.75
    simulation = LossSimulation(
        model="ghosh",
        direct=direct,
        indirect=total - direct,
        total=total,
        monetary_unit="billion_usd",
        time_basis="per_year",
    )

    selected = summarize_economy(
        simulation,
        sectors=["A"],
    )

    assert selected.loc["total", "mean"] == 3.0
