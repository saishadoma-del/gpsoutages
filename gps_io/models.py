# Role: Mathematical models for disruption losses
# Author: Saisha Doma
# Description: Implements Leontief, Ghosh, and Inoperability Input-Output Model
# calculations with explicit sector alignment and positive loss magnitudes.

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
from numpy.linalg import solve

from gps_io.data import TechnologyTables


@dataclass(frozen=True)
class InputOutputModel:
    direct_requirements: pd.DataFrame
    gross_output: pd.Series
    final_demand: pd.Series
    value_added: pd.Series
    monetary_unit: str = "billion_usd"
    time_basis: str = "per_year"

    @classmethod
    def from_tables(cls, tables: TechnologyTables) -> InputOutputModel:
        tables.validate()
        return cls(
            direct_requirements=tables.direct_requirements.copy(),
            gross_output=tables.gross_output.copy(),
            final_demand=tables.total_final_demand,
            value_added=tables.total_value_added,
            monetary_unit=tables.monetary_unit,
        )

    def __post_init__(self) -> None:
        sectors = list(self.direct_requirements.index)
        if list(self.direct_requirements.columns) != sectors:
            raise ValueError("Direct requirements rows and columns are not aligned")
        for name, vector in (
            ("gross output", self.gross_output),
            ("final demand", self.final_demand),
            ("value added", self.value_added),
        ):
            if list(vector.index) != sectors:
                raise ValueError(f"{name} sector IDs or order do not match the matrix")
            if not np.isfinite(vector.to_numpy(float)).all():
                raise ValueError(f"{name} contains missing or nonfinite values")
        if not np.isfinite(self.direct_requirements.to_numpy(float)).all():
            raise ValueError("Direct requirements contains missing or nonfinite values")
        if (self.gross_output <= 0).any():
            raise ValueError("Gross output must be positive")

    @property
    def sectors(self) -> tuple[str, ...]:
        return tuple(self.direct_requirements.index)

    @cached_property
    def leontief_inverse(self) -> pd.DataFrame:
        matrix = self.direct_requirements.to_numpy(float)
        identity = np.eye(len(self.sectors))
        inverse = solve(identity - matrix, identity)
        return pd.DataFrame(inverse, index=self.sectors, columns=self.sectors)

    @cached_property
    def allocation_coefficients(self) -> pd.DataFrame:
        output = self.gross_output.to_numpy(float)
        requirements = self.direct_requirements.to_numpy(float)
        coefficients = np.diag(1 / output) @ requirements @ np.diag(output)
        return pd.DataFrame(coefficients, index=self.sectors, columns=self.sectors)

    @cached_property
    def ghosh_inverse(self) -> pd.DataFrame:
        coefficients = self.allocation_coefficients.to_numpy(float)
        identity = np.eye(len(self.sectors))
        inverse = solve(identity - coefficients, identity)
        return pd.DataFrame(inverse, index=self.sectors, columns=self.sectors)

    @cached_property
    def inoperability_inverse(self) -> pd.DataFrame:
        coefficients = self.allocation_coefficients.to_numpy(float)
        identity = np.eye(len(self.sectors))
        inverse = solve(identity - coefficients, identity)
        return pd.DataFrame(inverse, index=self.sectors, columns=self.sectors)

    def _aligned_loss(self, direct_loss: pd.Series) -> np.ndarray:
        missing = sorted(set(self.sectors) - set(direct_loss.index))
        extra = sorted(set(direct_loss.index) - set(self.sectors))
        if missing or extra:
            raise ValueError(
                f"Direct loss sector IDs do not match the model. "
                f"Missing: {missing}. Extra: {extra}"
            )
        aligned = direct_loss.reindex(self.sectors).to_numpy(float)
        if not np.isfinite(aligned).all():
            raise ValueError("Direct loss contains missing or nonfinite values")
        if (aligned < 0).any():
            raise ValueError("Direct loss must use nonnegative loss magnitudes")
        return aligned

    def leontief_loss_from_final_demand(
        self,
        direct_loss: pd.Series,
    ) -> pd.Series:
        aligned = self._aligned_loss(direct_loss)
        total = self.leontief_inverse.to_numpy(float) @ aligned
        return pd.Series(total, index=self.sectors, name="total_loss")

    def ghosh_loss_from_value_added(
        self,
        direct_loss: pd.Series,
    ) -> pd.Series:
        aligned = self._aligned_loss(direct_loss)
        total = aligned @ self.ghosh_inverse.to_numpy(float)
        return pd.Series(total, index=self.sectors, name="total_loss")

    def inoperability_loss(
        self,
        direct_loss: pd.Series,
    ) -> pd.Series:
        aligned = self._aligned_loss(direct_loss)
        output = self.gross_output.to_numpy(float)
        direct_inoperability = aligned / output
        total_inoperability = (
            self.inoperability_inverse.to_numpy(float) @ direct_inoperability
        )
        total = total_inoperability * output
        return pd.Series(total, index=self.sectors, name="total_loss")
