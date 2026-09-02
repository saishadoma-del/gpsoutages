# Role: Regression test for the random draw order in the source notebook
# Author: Saisha Doma
# Description: Preserves the draw by draw random number assignment used for the
# submitted 10,000 sample results.

import numpy as np

from gps_io.simulation import draw_dependency_fractions


def test_dependency_draw_order_matches_source_notebook() -> None:
    sectors = ("A", "B", "C")
    ranges = {
        "A": (0.1, 0.2),
        "B": (0.3, 0.4),
        "C": (0.5, 0.6),
    }
    seed = 42
    draw_count = 8
    generator = np.random.default_rng(seed)
    expected = np.array(
        [
            [
                generator.uniform(*ranges[sector])
                for sector in sectors
            ]
            for _ in range(draw_count)
        ]
    )

    actual = draw_dependency_fractions(
        sectors,
        ranges,
        draws=draw_count,
        seed=seed,
    )

    np.testing.assert_allclose(actual.to_numpy(), expected)
