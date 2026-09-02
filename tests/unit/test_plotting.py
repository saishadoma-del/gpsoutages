# Role: Unit tests for publication plotting
# Author: Saisha Doma
# Description: Verifies compact panels, requested typography, ranges, and output.

import re

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from gps_io.plotting import _format_panel_statistics
from gps_io.plotting import BAR_HEIGHT_AXIS_UNITS
from gps_io.plotting import build_duration_figure
from gps_io.plotting import FIGURE_DIMENSIONS_INCHES
from gps_io.plotting import FIGURE_DPI
from gps_io.plotting import plot_duration_panels
from gps_io.simulation import LossSimulation


def _anti_correlated_simulation() -> LossSimulation:
    total = pd.DataFrame(
        [
            [1_000.0, 9_000.0],
            [9_000.0, 1_000.0],
            [2_000.0, 8_000.0],
            [8_000.0, 2_000.0],
        ],
        columns=["A", "B"],
    )
    direct = total * 0.75
    return LossSimulation(
        model="leontief",
        direct=direct,
        indirect=total - direct,
        total=total,
        monetary_unit="billion_usd",
        time_basis="per_year",
    )


def test_duration_figure_uses_compact_shared_panels() -> None:
    figure = build_duration_figure(
        _anti_correlated_simulation(),
        price_year=2023,
        sector_labels={"A": "Alpha", "B": "Beta"},
    )
    try:
        assert len(figure.axes) == 4
        assert tuple(figure.get_size_inches()) == FIGURE_DIMENSIONS_INCHES[
            "all_sectors"
        ]
        assert [axis.get_title(loc="left") for axis in figure.axes] == [
            "(a) 1-hour outage",
            "(b) 6-hour outage",
            "(c) 12-hour outage",
            "(d) 24-hour outage",
        ]
        assert all(
            axis.get_title(loc=location) == ""
            for axis in figure.axes
            for location in ("center", "right")
        )
        assert figure._suptitle is None
        assert (
            figure._supxlabel.get_text()
            == "Economic loss (billion 2023 USD per outage)"
        )
        assert all(
            axis.get_xlim() == figure.axes[0].get_xlim()
            for axis in figure.axes
        )
        assert all(
            any(line.get_visible() for line in axis.get_xgridlines())
            for axis in figure.axes
        )
        assert all(
            any(line.get_visible() for line in axis.get_ygridlines())
            for axis in figure.axes
        )
        assert all(
            axis.texts[0].get_fontfamily() == ["Times New Roman"]
            for axis in figure.axes
        )
        assert all(
            axis.texts[0].get_bbox_patch() is None
            for axis in figure.axes
        )
        assert figure.legends[0].get_frame_on() is False
    finally:
        plt.close(figure)


def test_panel_annotations_use_ordered_draw_total_intervals() -> None:
    figure = build_duration_figure(
        _anti_correlated_simulation(),
        price_year=2023,
        sector_labels={"A": "Alpha", "B": "Beta"},
    )
    try:
        for axis in figure.axes:
            annotation = axis.texts[0].get_text()
            match = re.search(
                r"95% draw interval: \$(\d+\.\d{2})B to \$(\d+\.\d{2})B",
                annotation,
            )
            assert match is not None
            low, high = (float(value) for value in match.groups())
            assert low <= high
            assert low == high
    finally:
        plt.close(figure)


def test_panel_statistics_reject_reversed_interval() -> None:
    summary = pd.DataFrame(
        {
            "low_2.5pct": [0.0, 0.0, 3.0],
            "mean": [1.0, 1.0, 2.0],
            "high_97.5pct": [2.0, 2.0, 1.0],
        },
        index=["direct", "indirect", "total"],
    )

    with pytest.raises(ValueError, match="low <= mean <= high"):
        _format_panel_statistics(summary)


def test_plot_duration_panels_writes_and_closes_figure(tmp_path) -> None:
    open_figures = set(plt.get_fignums())
    destination = tmp_path / "figure.png"

    plot_duration_panels(
        _anti_correlated_simulation(),
        destination,
        price_year=2023,
        sector_labels={"A": "Alpha", "B": "Beta"},
    )

    assert destination.stat().st_size > 0
    image = plt.imread(destination)
    width, height = FIGURE_DIMENSIONS_INCHES["all_sectors"]
    assert image.shape[:2] == (
        round(height * FIGURE_DPI),
        round(width * FIGURE_DPI),
    )
    assert set(plt.get_fignums()) == open_figures


def test_selected_sector_canvas_matches_full_figure_bar_thickness() -> None:
    columns = [f"S{index}" for index in range(13)]
    total = pd.DataFrame(
        [
            [(index + 1) * multiplier for index in range(13)]
            for multiplier in (0.8, 1.0, 1.2, 1.4)
        ],
        columns=columns,
    )
    simulation = LossSimulation(
        model="ghosh",
        direct=total * 0.6,
        indirect=total * 0.4,
        total=total,
        monetary_unit="billion_usd",
        time_basis="per_year",
    )
    labels = {sector: sector for sector in columns}
    figures = [
        build_duration_figure(
            simulation,
            price_year=2023,
            sector_labels=labels,
        ),
        build_duration_figure(
            simulation,
            price_year=2023,
            selected_sectors=tuple(columns[:3]),
            sector_labels=labels,
        ),
    ]
    try:
        assert tuple(figures[1].get_size_inches()) == (
            FIGURE_DIMENSIONS_INCHES["selected_sectors"]
        )
        thicknesses = []
        for figure in figures:
            figure.canvas.draw()
            axis = figure.axes[0]
            patch = axis.patches[0]
            bottom = axis.transData.transform((0, patch.get_y()))[1]
            top = axis.transData.transform(
                (0, patch.get_y() + BAR_HEIGHT_AXIS_UNITS)
            )[1]
            thicknesses.append(abs(top - bottom) / figure.dpi)
        assert abs(thicknesses[0] - thicknesses[1]) < 0.01
    finally:
        for figure in figures:
            plt.close(figure)
