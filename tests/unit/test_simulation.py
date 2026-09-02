# Role: Unit tests for dependency draws and matched model simulation
# Author: Saisha Doma
# Description: Verifies deterministic draws, validation, and shared inputs.

import pandas as pd
import pytest

from gps_io.models import InputOutputModel
from gps_io.simulation import draw_dependency_fractions
from gps_io.simulation import simulate_all_models


def test_dependency_draws_are_deterministic(two_sector_model: InputOutputModel) -> None:
    ranges = {"A": (0.1, 0.2), "B": (0.3, 0.4)}

    first = draw_dependency_fractions(
        two_sector_model.sectors,
        ranges,
        draws=12,
        seed=9,
    )
    second = draw_dependency_fractions(
        two_sector_model.sectors,
        ranges,
        draws=12,
        seed=9,
    )

    pd.testing.assert_frame_equal(first, second)


def test_all_models_use_the_same_direct_dependency_draws(
    two_sector_model: InputOutputModel,
) -> None:
    ranges = {"A": (0.1, 0.2), "B": (0.3, 0.4)}
    draws = draw_dependency_fractions(
        two_sector_model.sectors,
        ranges,
        draws=12,
        seed=9,
    )

    simulations = simulate_all_models(two_sector_model, draws)

    pd.testing.assert_frame_equal(
        simulations["ghosh"].direct,
        simulations["iim"].direct,
    )
    assert list(simulations) == ["leontief", "ghosh", "iim"]


def test_dependency_range_ids_must_match_model(
    two_sector_model: InputOutputModel,
) -> None:
    with pytest.raises(ValueError, match="Missing.*B.*Extra.*C"):
        draw_dependency_fractions(
            two_sector_model.sectors,
            {"A": (0.1, 0.2), "C": (0.3, 0.4)},
        )
