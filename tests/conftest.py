# Role: Shared compact fixtures for the GPS input-output test suite
# Author: Saisha Doma
# Description: Provides a two sector model with a nonsymmetric coefficient matrix.

import pandas as pd
import pytest

from gps_io.data import TechnologyTables
from gps_io.models import InputOutputModel


@pytest.fixture
def two_sector_model() -> InputOutputModel:
    sectors = ["A", "B"]
    tables = TechnologyTables(
        direct_requirements=pd.DataFrame(
            [[0.10, 0.20], [0.05, 0.10]],
            index=sectors,
            columns=sectors,
        ),
        gross_output=pd.Series([100.0, 200.0], index=sectors),
        value_added=pd.DataFrame(
            [[40.0, 80.0]],
            index=["value_added"],
            columns=sectors,
        ),
        final_demand=pd.DataFrame(
            [[50.0], [100.0]],
            index=sectors,
            columns=["final_demand"],
        ),
        monetary_unit="billion_usd",
    )
    return InputOutputModel.from_tables(tables)
