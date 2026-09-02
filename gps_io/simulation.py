# Role: Monte Carlo simulation for GPS disruption scenarios
# Author: Saisha Doma
# Description: Generates shared dependency draws and applies each input-output model
# with deterministic seeds and consistent positive loss magnitudes.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from typing import Mapping

import numpy as np
import pandas as pd

from gps_io.constants import HOURS_PER_YEAR
from gps_io.models import InputOutputModel

ModelKind = Literal["leontief", "ghosh", "iim"]


@dataclass(frozen=True)
class LossSimulation:
    model: ModelKind
    direct: pd.DataFrame
    indirect: pd.DataFrame
    total: pd.DataFrame
    monetary_unit: str
    time_basis: str

    def validate(self) -> None:
        if list(self.direct.columns) != list(self.indirect.columns):
            raise ValueError("Direct and indirect result sectors do not match")
        if list(self.direct.columns) != list(self.total.columns):
            raise ValueError("Direct and total result sectors do not match")
        if not self.direct.index.equals(self.indirect.index):
            raise ValueError("Direct and indirect draw IDs do not match")
        if not self.direct.index.equals(self.total.index):
            raise ValueError("Direct and total draw IDs do not match")
        for name, values in (
            ("direct", self.direct),
            ("indirect", self.indirect),
            ("total", self.total),
        ):
            if not np.isfinite(values.to_numpy(float)).all():
                raise ValueError(f"{name} simulation results contain nonfinite values")


def draw_dependency_fractions(
    sectors: tuple[str, ...] | list[str],
    dependency_ranges: Mapping[str, tuple[float, float]],
    *,
    draws: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    sector_ids = list(sectors)
    if draws <= 0:
        raise ValueError("Draw count must be positive")
    missing = sorted(set(sector_ids) - set(dependency_ranges))
    extra = sorted(set(dependency_ranges) - set(sector_ids))
    if missing or extra:
        raise ValueError(
            f"Dependency range sector IDs do not match the model. "
            f"Missing: {missing}. Extra: {extra}"
        )

    bounds: list[tuple[float, float]] = []
    for sector in sector_ids:
        lower, upper = dependency_ranges[sector]
        if not 0 <= lower <= upper <= 1:
            raise ValueError(
                f"Dependency range for {sector} must satisfy 0 <= lower <= upper <= 1"
            )
        bounds.append((lower, upper))

    lower_bounds = np.array([lower for lower, _ in bounds])
    upper_bounds = np.array([upper for _, upper in bounds])
    generator = np.random.default_rng(seed)
    values = generator.uniform(
        lower_bounds,
        upper_bounds,
        size=(draws, len(sector_ids)),
    )

    return pd.DataFrame(
        values,
        index=pd.RangeIndex(draws, name="draw"),
        columns=sector_ids,
    )


def _validate_dependency_draws(
    model: InputOutputModel,
    dependency_draws: pd.DataFrame,
) -> np.ndarray:
    if list(dependency_draws.columns) != list(model.sectors):
        raise ValueError(
            "Dependency draw columns must match model sector IDs and order"
        )
    values = dependency_draws.to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Dependency draws contain missing or nonfinite values")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("Dependency draws must be within [0, 1]")
    return values


def simulate_model(
    model: InputOutputModel,
    model_kind: ModelKind,
    dependency_draws: pd.DataFrame,
) -> LossSimulation:
    fractions = _validate_dependency_draws(model, dependency_draws)

    if model_kind == "leontief":
        base = model.final_demand.to_numpy(float)
        direct_values = fractions * base
        total_values = (
            direct_values @ model.leontief_inverse.to_numpy(float).T
        )
    elif model_kind == "ghosh":
        base = model.value_added.to_numpy(float)
        direct_values = fractions * base
        total_values = direct_values @ model.ghosh_inverse.to_numpy(float)
    elif model_kind == "iim":
        base = model.value_added.to_numpy(float)
        direct_values = fractions * base
        output = model.gross_output.to_numpy(float)
        direct_inoperability = direct_values / output
        total_inoperability = (
            direct_inoperability
            @ model.inoperability_inverse.to_numpy(float).T
        )
        total_values = total_inoperability * output
    else:
        raise ValueError(f"Unsupported model kind: {model_kind}")

    columns = list(model.sectors)
    index = dependency_draws.index.copy()
    direct = pd.DataFrame(direct_values, index=index, columns=columns)
    total = pd.DataFrame(total_values, index=index, columns=columns)
    indirect = total - direct
    simulation = LossSimulation(
        model=model_kind,
        direct=direct,
        indirect=indirect,
        total=total,
        monetary_unit=model.monetary_unit,
        time_basis=model.time_basis,
    )
    simulation.validate()
    return simulation


def simulate_all_models(
    model: InputOutputModel,
    dependency_draws: pd.DataFrame,
) -> dict[ModelKind, LossSimulation]:
    return {
        model_kind: simulate_model(model, model_kind, dependency_draws)
        for model_kind in ("leontief", "ghosh", "iim")
    }


def scale_to_outage(
    simulation: LossSimulation,
    hours: int | float,
) -> LossSimulation:
    if not 0 < hours <= HOURS_PER_YEAR:
        raise ValueError(
            f"Outage duration must be within (0, {HOURS_PER_YEAR}] hours"
        )
    if simulation.time_basis != "per_year":
        raise ValueError(
            f"Expected annual input results, received {simulation.time_basis!r}"
        )
    factor = float(hours) / HOURS_PER_YEAR
    scaled = LossSimulation(
        model=simulation.model,
        direct=simulation.direct * factor,
        indirect=simulation.indirect * factor,
        total=simulation.total * factor,
        monetary_unit=simulation.monetary_unit,
        time_basis=f"per_{hours:g}_hour_outage",
    )
    scaled.validate()
    return scaled
