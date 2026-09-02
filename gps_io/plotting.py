# Role: Figures for GPS disruption loss simulations
# Author: Saisha Doma
# Description: Creates publication figures with draw-based uncertainty intervals
# and a compact panel layout adapted from C-SWIM visualization conventions.

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from gps_io.constants import OUTAGE_HOURS
from gps_io.constants import SECTOR_LABELS
from gps_io.simulation import LossSimulation
from gps_io.simulation import scale_to_outage
from gps_io.summaries import summarize_economy
from gps_io.summaries import summarize_sectors

PAPER_FIGURE_NAMES = {
    "leontief": "Leontief_Model.png",
    "ghosh": "Ghosh_Model.png",
    "iim": "Inoperability_IO_Model.png",
}

PAPER_SELECTED_FIGURE_NAMES = {
    "ghosh": "Ghosh_Model_S.png",
    "iim": "Inoperability_IO_Model_S (1).png",
}


DIRECT_COLOR = "darkred"
INDIRECT_COLOR = "lightcoral"
BACKGROUND_COLOR = "#F0F0F0"
PANEL_LABELS = ("a", "b", "c", "d")
FIGURE_DIMENSIONS_INCHES = {
    "all_sectors": (8.0, 7.7),
    "selected_sectors": (8.0, 3.3),
}
FIGURE_DPI = 300
BAR_HEIGHT_AXIS_UNITS = 0.72
PLOT_RC_PARAMS = {
    "font.family": "Times New Roman",
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _format_panel_statistics(economy: pd.DataFrame) -> str:
    total = economy.loc["total"]
    low = float(total["low_2.5pct"])
    mean = float(total["mean"])
    high = float(total["high_97.5pct"])
    if not low <= mean <= high:
        raise ValueError(
            "Economy interval must be ordered as low <= mean <= high"
        )
    return (
        f"Direct: ${economy.loc['direct', 'mean']:.2f}B\n"
        f"Indirect: ${economy.loc['indirect', 'mean']:.2f}B\n"
        f"Total: ${mean:.2f}B\n"
        f"95% draw interval: ${low:.2f}B to ${high:.2f}B"
    )


def build_duration_figure(
    annual_simulation: LossSimulation,
    *,
    price_year: int,
    selected_sectors: tuple[str, ...] | None = None,
    sector_labels: Mapping[str, str] = SECTOR_LABELS,
) -> Figure:
    """Build a four-panel outage-duration figure without saving it."""
    annual_simulation.validate()
    sectors = list(annual_simulation.total.columns)
    if selected_sectors is not None:
        missing = sorted(set(selected_sectors) - set(sectors))
        if missing:
            raise ValueError(f"Selected sectors do not exist: {missing}")
        sectors = list(selected_sectors)

    missing_labels = sorted(set(sectors) - set(sector_labels))
    if missing_labels:
        raise ValueError(f"Sector labels do not exist: {missing_labels}")

    annual_total_mean = annual_simulation.total[sectors].mean()
    sector_order = annual_total_mean.sort_values(ascending=False).index.tolist()
    labels = [sector_labels[sector] for sector in sector_order]
    duration_simulations = {
        hours: scale_to_outage(annual_simulation, hours)
        for hours in OUTAGE_HOURS
    }
    max_upper = max(
        summarize_sectors(simulation)["total"]
        .loc[sector_order, "high_97.5pct"]
        .max()
        for simulation in duration_simulations.values()
    )
    x_limit = max(float(max_upper) * 1.12, 1e-9)
    baseline_gap = x_limit * 0.012
    figure_kind = (
        "all_sectors" if selected_sectors is None else "selected_sectors"
    )
    figure_size = FIGURE_DIMENSIONS_INCHES[figure_kind]
    layout_bottom = 0.075 if selected_sectors is None else 0.18
    xlabel_y = 0.045 if selected_sectors is None else 0.095

    with mpl.rc_context(PLOT_RC_PARAMS):
        figure, axes = plt.subplots(
            2,
            2,
            figsize=figure_size,
            sharex=True,
            sharey=True,
        )
        positions = np.arange(len(sector_order))[::-1]

        for panel, hours in enumerate(OUTAGE_HOURS):
            row, column = divmod(panel, 2)
            axis = axes[row, column]
            simulation = duration_simulations[hours]
            summaries = summarize_sectors(simulation)
            economy = summarize_economy(simulation, sectors=sector_order)

            direct = summaries["direct"].loc[sector_order, "mean"]
            indirect = summaries["indirect"].loc[sector_order, "mean"]
            total = summaries["total"].loc[sector_order]
            lower_error = np.maximum(
                total["mean"] - total["low_2.5pct"],
                0,
            )
            upper_error = np.maximum(
                total["high_97.5pct"] - total["mean"],
                0,
            )

            axis.barh(
                positions,
                direct,
                left=baseline_gap,
                color=DIRECT_COLOR,
                alpha=0.8,
                height=BAR_HEIGHT_AXIS_UNITS,
                zorder=3,
            )
            axis.barh(
                positions,
                indirect,
                left=direct + baseline_gap,
                color=INDIRECT_COLOR,
                alpha=0.8,
                height=BAR_HEIGHT_AXIS_UNITS,
                zorder=3,
            )
            axis.errorbar(
                total["mean"] + baseline_gap,
                positions,
                xerr=[lower_error, upper_error],
                fmt="none",
                color="#222222",
                capsize=2,
                capthick=0.8,
                elinewidth=0.8,
                zorder=4,
            )

            axis.set_title(
                f"({PANEL_LABELS[panel]}) {hours:g}-hour outage",
                loc="left",
                fontsize=10,
                fontweight="bold",
                pad=6,
            )
            axis.text(
                0.98,
                0.04,
                _format_panel_statistics(economy),
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.6,
                parse_math=False,
                linespacing=1.25,
                zorder=6,
            )

            axis.set_xlim(0, x_limit)
            axis.set_ylim(-0.6, len(sector_order) - 0.4)
            axis.grid(
                True,
                axis="x",
                linewidth=0.45,
                color="#C8C8C8",
                zorder=0,
            )
            axis.grid(
                True,
                axis="y",
                linewidth=0.3,
                color="#D4D4D4",
                zorder=0,
            )
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_color("#555555")
            axis.spines["bottom"].set_color("#777777")
            axis.spines["left"].set_linewidth(0.7)
            axis.spines["bottom"].set_linewidth(0.7)
            axis.tick_params(axis="both", labelsize=8, colors="#333333")
            axis.set_yticks(positions)
            if column == 0:
                axis.set_yticklabels(labels, fontsize=8)
            else:
                axis.tick_params(axis="y", labelleft=False, left=False)
            if row == 0:
                axis.tick_params(axis="x", labelbottom=False)
            axis.set_facecolor(BACKGROUND_COLOR)

        legend_handles = [
            Patch(facecolor=DIRECT_COLOR, label="Direct loss"),
            Patch(facecolor=INDIRECT_COLOR, label="Indirect loss"),
            Line2D(
                [0],
                [0],
                color="#222222",
                linewidth=0.8,
                marker="|",
                markersize=6,
                label="Sector 95% draw interval",
            ),
        ]
        figure.supxlabel(
            f"Economic loss (billion {price_year} USD per outage)",
            fontsize=10,
            y=xlabel_y,
        )
        figure.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=3,
            frameon=False,
            fontsize=8.5,
        )
        figure.patch.set_facecolor(BACKGROUND_COLOR)
        figure.tight_layout(
            rect=(0, layout_bottom, 1, 0.995),
            h_pad=0.8,
            w_pad=0.3,
        )

    return figure


def plot_duration_panels(
    annual_simulation: LossSimulation,
    output_path: str | Path,
    *,
    price_year: int,
    selected_sectors: tuple[str, ...] | None = None,
    sector_labels: Mapping[str, str] = SECTOR_LABELS,
) -> None:
    """Save a compact duration figure and close its Matplotlib resources."""
    figure = build_duration_figure(
        annual_simulation,
        price_year=price_year,
        selected_sectors=selected_sectors,
        sector_labels=sector_labels,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(PLOT_RC_PARAMS):
        figure.savefig(
            destination,
            dpi=FIGURE_DPI,
            facecolor=figure.get_facecolor(),
        )
    plt.close(figure)


