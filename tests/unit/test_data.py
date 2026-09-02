# Role: Unit tests for BEA data aggregation and schema checks
# Author: Saisha Doma
# Description: Verifies the 13 sector concordance, units, and invalid value handling.

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gps_io.constants import FINAL_DEMAND_COLUMNS
from gps_io.constants import SECTOR_GROUPS
from gps_io.constants import SECTOR_IDS
from gps_io.constants import TOTAL_INDUSTRY_OUTPUT_ROW
from gps_io.constants import VALUE_ADDED_ROWS
from gps_io.data import build_technology_tables
from gps_io.data import load_bea_use_table


def _compact_use_table() -> pd.DataFrame:
    codes = [
        code
        for sector_codes in SECTOR_GROUPS.values()
        for code in sector_codes
    ]
    rows = codes + list(VALUE_ADDED_ROWS) + [TOTAL_INDUSTRY_OUTPUT_ROW]
    columns = codes + list(FINAL_DEMAND_COLUMNS)
    table = pd.DataFrame(0.0, index=rows, columns=columns)
    for code in codes:
        table.loc[code, code] = 10.0
        table.loc[TOTAL_INDUSTRY_OUTPUT_ROW, code] = 100.0
        table.loc[VALUE_ADDED_ROWS[0], code] = 40.0
        table.loc[code, FINAL_DEMAND_COLUMNS[0]] = 50.0
    return table


def test_aggregation_uses_canonical_13_sector_order() -> None:
    tables = build_technology_tables(_compact_use_table())

    assert tables.sectors == SECTOR_IDS
    assert tables.direct_requirements.shape == (13, 13)
    assert tables.direct_requirements.loc["TRADE", "TRADE"] == pytest.approx(0.1)
    assert tables.gross_output.loc["TRADE"] == pytest.approx(200.0)
    assert tables.total_final_demand.loc["TRADE"] == pytest.approx(100.0)


def test_loader_converts_million_to_billion(tmp_path: Path) -> None:
    path = tmp_path / "use_table.csv"
    pd.DataFrame({"11": ["1000"]}, index=["11"]).to_csv(path)

    loaded = load_bea_use_table(path, source_unit="million_usd")

    assert loaded.loc["11", "11"] == pytest.approx(1.0)


def test_loader_rejects_unexpected_nonnumeric_value(tmp_path: Path) -> None:
    path = tmp_path / "use_table.csv"
    pd.DataFrame({"11": ["unknown"]}, index=["11"]).to_csv(path)

    with pytest.raises(ValueError, match="nonnumeric value"):
        load_bea_use_table(path)


def test_aggregation_rejects_missing_required_code() -> None:
    table = _compact_use_table().drop(index="11")

    with pytest.raises(ValueError, match="missing required rows"):
        build_technology_tables(table)


def test_aggregated_values_are_finite() -> None:
    tables = build_technology_tables(_compact_use_table())

    assert np.isfinite(tables.direct_requirements.to_numpy()).all()
