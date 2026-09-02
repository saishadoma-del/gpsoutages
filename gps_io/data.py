# Role: Input and schema layer for BEA technology tables
# Author: Saisha Doma
# Description: Loads the compact BEA Use Table, performs the 13 sector aggregation,
# and reads or writes canonical model inputs with validation.

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

from gps_io.constants import FINAL_DEMAND_COLUMNS
from gps_io.constants import SECTOR_GROUPS
from gps_io.constants import SECTOR_IDS
from gps_io.constants import TOTAL_INDUSTRY_OUTPUT_ROW
from gps_io.constants import VALUE_ADDED_ROWS

UNIT_SCALE_TO_BILLION = {
    "million_usd": 1 / 1000,
    "billion_usd": 1.0,
}


@dataclass(frozen=True)
class TechnologyTables:
    direct_requirements: pd.DataFrame
    gross_output: pd.Series
    value_added: pd.DataFrame
    final_demand: pd.DataFrame
    intermediate_transactions: pd.DataFrame | None = None
    monetary_unit: str = "billion_usd"
    price_year: int | None = None
    table_vintage: int | None = None

    @property
    def sectors(self) -> tuple[str, ...]:
        return tuple(self.direct_requirements.index)

    @property
    def total_value_added(self) -> pd.Series:
        return self.value_added.sum(axis=0).rename("value_added")

    @property
    def total_final_demand(self) -> pd.Series:
        return self.final_demand.sum(axis=1).rename("final_demand")

    def validate(self) -> None:
        sectors = list(self.direct_requirements.index)
        if not sectors:
            raise ValueError("Direct requirements matrix has no sectors")
        if not self.direct_requirements.index.is_unique:
            raise ValueError("Direct requirements matrix has duplicate row sector IDs")
        if not self.direct_requirements.columns.is_unique:
            raise ValueError("Direct requirements matrix has duplicate column sector IDs")
        if list(self.direct_requirements.columns) != sectors:
            raise ValueError(
                "Direct requirements rows and columns must have identical sector IDs and order"
            )
        if list(self.gross_output.index) != sectors:
            raise ValueError("Gross output sector IDs or order do not match the matrix")
        if list(self.value_added.columns) != sectors:
            raise ValueError("Value added sector IDs or order do not match the matrix")
        if list(self.final_demand.index) != sectors:
            raise ValueError("Final demand sector IDs or order do not match the matrix")
        if self.intermediate_transactions is not None:
            if list(self.intermediate_transactions.index) != sectors:
                raise ValueError("Intermediate transaction row IDs do not match the matrix")
            if list(self.intermediate_transactions.columns) != sectors:
                raise ValueError("Intermediate transaction column IDs do not match the matrix")
        _require_finite("direct requirements", self.direct_requirements.to_numpy(float))
        _require_finite("gross output", self.gross_output.to_numpy(float))
        _require_finite("value added", self.value_added.to_numpy(float))
        _require_finite("final demand", self.final_demand.to_numpy(float))
        if (self.gross_output <= 0).any():
            invalid = self.gross_output[self.gross_output <= 0].index.tolist()
            raise ValueError(f"Gross output must be positive for sectors {invalid}")
        if (self.direct_requirements < 0).any().any():
            raise ValueError("Direct requirements matrix contains negative coefficients")


def _require_finite(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains missing or nonfinite values")


def _industry_codes() -> tuple[str, ...]:
    return tuple(
        code
        for sector_codes in SECTOR_GROUPS.values()
        for code in sector_codes
    )


def _code_to_sector() -> dict[str, str]:
    return {
        code: sector
        for sector, sector_codes in SECTOR_GROUPS.items()
        for code in sector_codes
    }


def load_bea_use_table(
    path: str | Path,
    *,
    source_unit: str = "million_usd",
) -> pd.DataFrame:
    source_path = Path(path)
    if source_unit not in UNIT_SCALE_TO_BILLION:
        raise ValueError(
            f"Unsupported monetary unit {source_unit!r}. "
            f"Expected one of {sorted(UNIT_SCALE_TO_BILLION)}"
        )
    if not source_path.is_file():
        raise FileNotFoundError(f"BEA Use Table does not exist: {source_path}")

    raw = pd.read_csv(source_path, index_col=0, dtype=str)
    raw.index = raw.index.map(str)
    raw.columns = raw.columns.map(str)
    if not raw.index.is_unique:
        duplicates = raw.index[raw.index.duplicated()].unique().tolist()
        raise ValueError(f"BEA Use Table has duplicate row codes: {duplicates}")
    if not raw.columns.is_unique:
        duplicates = raw.columns[raw.columns.duplicated()].unique().tolist()
        raise ValueError(f"BEA Use Table has duplicate column codes: {duplicates}")

    stripped = raw.apply(lambda column: column.str.strip())
    explicit_zero = stripped.eq("---")
    numeric = stripped.mask(explicit_zero, "0").apply(pd.to_numeric, errors="coerce")
    invalid = numeric.isna() & stripped.notna() & stripped.ne("")
    if invalid.any().any():
        row_code, column_code = invalid.stack().loc[lambda values: values].index[0]
        value = raw.loc[row_code, column_code]
        raise ValueError(
            f"BEA Use Table {source_path} has nonnumeric value {value!r} "
            f"at row {row_code!r}, column {column_code!r}"
        )
    if numeric.isna().any().any():
        row_code, column_code = numeric.isna().stack().loc[lambda values: values].index[0]
        raise ValueError(
            f"BEA Use Table {source_path} has a missing value "
            f"at row {row_code!r}, column {column_code!r}"
        )

    return numeric * UNIT_SCALE_TO_BILLION[source_unit]


def build_technology_tables(
    use_table: pd.DataFrame,
    *,
    price_year: int | None = None,
    table_vintage: int | None = None,
) -> TechnologyTables:
    codes = _industry_codes()
    code_to_sector = _code_to_sector()
    required_rows = set(codes) | set(VALUE_ADDED_ROWS) | {TOTAL_INDUSTRY_OUTPUT_ROW}
    required_columns = set(codes) | set(FINAL_DEMAND_COLUMNS)
    missing_rows = sorted(required_rows - set(use_table.index))
    missing_columns = sorted(required_columns - set(use_table.columns))
    if missing_rows:
        raise ValueError(f"BEA Use Table is missing required rows: {missing_rows}")
    if missing_columns:
        raise ValueError(f"BEA Use Table is missing required columns: {missing_columns}")

    intermediate = use_table.loc[list(codes), list(codes)]
    grouped_rows = (
        intermediate.rename(index=code_to_sector)
        .groupby(level=0, sort=False)
        .sum()
    )
    transactions = (
        grouped_rows.T.rename(index=code_to_sector)
        .groupby(level=0, sort=False)
        .sum()
        .T.reindex(index=SECTOR_IDS, columns=SECTOR_IDS)
    )

    gross_by_code = use_table.loc[TOTAL_INDUSTRY_OUTPUT_ROW, list(codes)].copy()
    gross_by_code.index = gross_by_code.index.map(code_to_sector)
    gross_output = (
        gross_by_code.groupby(level=0, sort=False)
        .sum()
        .reindex(SECTOR_IDS)
        .rename("gross_output")
    )

    direct_requirements = transactions.div(gross_output, axis="columns").round(6)

    value_added_by_code = use_table.loc[list(VALUE_ADDED_ROWS), list(codes)]
    value_added = (
        value_added_by_code.rename(columns=code_to_sector)
        .T.groupby(level=0, sort=False)
        .sum()
        .T.reindex(columns=SECTOR_IDS)
    )

    final_demand_by_code = use_table.loc[list(codes), list(FINAL_DEMAND_COLUMNS)]
    final_demand = (
        final_demand_by_code.rename(index=code_to_sector)
        .groupby(level=0, sort=False)
        .sum()
        .reindex(index=SECTOR_IDS)
    )

    tables = TechnologyTables(
        direct_requirements=direct_requirements,
        gross_output=gross_output,
        value_added=value_added,
        final_demand=final_demand,
        intermediate_transactions=transactions,
        monetary_unit="billion_usd",
        price_year=price_year,
        table_vintage=table_vintage,
    )
    tables.validate()
    return tables


def build_technology_tables_from_csv(
    path: str | Path,
    *,
    source_unit: str = "million_usd",
    price_year: int | None = None,
    table_vintage: int | None = None,
) -> TechnologyTables:
    use_table = load_bea_use_table(path, source_unit=source_unit)
    return build_technology_tables(
        use_table,
        price_year=price_year,
        table_vintage=table_vintage,
    )


def _atomic_to_csv(
    data: pd.DataFrame | pd.Series,
    path: Path,
    **kwargs: object,
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
        data.to_csv(temporary_path, **kwargs)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_technology_tables(
    tables: TechnologyTables,
    output_dir: str | Path,
    *,
    year_label: str,
) -> None:
    tables.validate()
    target = Path(output_dir)
    _atomic_to_csv(
        tables.direct_requirements,
        target / "direct_requirements.csv",
    )
    _atomic_to_csv(
        tables.gross_output.rename(year_label),
        target / "gross_output.csv",
        header=True,
    )
    _atomic_to_csv(
        tables.value_added,
        target / "value_added.csv",
    )
    _atomic_to_csv(
        tables.final_demand,
        target / "final_demand.csv",
    )
    _atomic_to_csv(
        tables.total_value_added.rename(year_label),
        target / "total_value_added.csv",
        header=True,
    )
    if tables.intermediate_transactions is not None:
        _atomic_to_csv(
            tables.intermediate_transactions.sum(axis=0).rename(year_label),
            target / "total_intermediate_use.csv",
            header=True,
        )


def load_technology_tables(
    input_dir: str | Path,
    *,
    monetary_unit: str = "billion_usd",
    price_year: int | None = None,
    table_vintage: int | None = None,
) -> TechnologyTables:
    source = Path(input_dir)
    required = {
        "direct requirements": source / "direct_requirements.csv",
        "gross output": source / "gross_output.csv",
        "value added": source / "value_added.csv",
        "final demand": source / "final_demand.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Technology table files do not exist: {missing}")

    direct_requirements = pd.read_csv(required["direct requirements"], index_col=0)
    gross_frame = pd.read_csv(required["gross output"], index_col=0)
    if gross_frame.shape[1] != 1:
        raise ValueError(
            f"Gross output file {required['gross output']} must have one value column"
        )
    gross_output = gross_frame.iloc[:, 0].rename("gross_output")
    value_added = pd.read_csv(required["value added"], index_col=0)
    final_demand = pd.read_csv(required["final demand"], index_col=0)

    tables = TechnologyTables(
        direct_requirements=direct_requirements,
        gross_output=gross_output,
        value_added=value_added,
        final_demand=final_demand,
        monetary_unit=monetary_unit,
        price_year=price_year,
        table_vintage=table_vintage,
    )
    tables.validate()
    return tables
