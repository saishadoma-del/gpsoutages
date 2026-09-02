# Role: Scientific tests for the three input-output formulations
# Author: Saisha Doma
# Description: Checks matrix orientation, sign convention, identity behavior,
# and sector alignment against direct NumPy calculations.

import numpy as np
import pandas as pd
import pytest
from numpy.linalg import solve

from gps_io.data import TechnologyTables
from gps_io.models import InputOutputModel


def test_leontief_matches_column_vector_formulation(two_sector_model: InputOutputModel) -> None:
    direct = pd.Series({"A": 3.0, "B": 5.0})
    matrix = two_sector_model.direct_requirements.to_numpy()
    expected = solve(np.eye(2) - matrix, direct.to_numpy())

    actual = two_sector_model.leontief_loss_from_final_demand(direct)

    np.testing.assert_allclose(actual.to_numpy(), expected)


def test_ghosh_matches_row_vector_formulation(two_sector_model: InputOutputModel) -> None:
    direct = pd.Series({"A": 3.0, "B": 5.0})
    matrix = two_sector_model.direct_requirements.to_numpy()
    output = two_sector_model.gross_output.to_numpy()
    allocation = np.diag(1 / output) @ matrix @ np.diag(output)
    inverse = solve(np.eye(2) - allocation, np.eye(2))
    expected = direct.to_numpy() @ inverse

    actual = two_sector_model.ghosh_loss_from_value_added(direct)

    np.testing.assert_allclose(actual.to_numpy(), expected)


def test_iim_matches_column_inoperability_formulation(
    two_sector_model: InputOutputModel,
) -> None:
    direct = pd.Series({"A": 3.0, "B": 5.0})
    matrix = two_sector_model.direct_requirements.to_numpy()
    output = two_sector_model.gross_output.to_numpy()
    allocation = np.diag(1 / output) @ matrix @ np.diag(output)
    inverse = solve(np.eye(2) - allocation, np.eye(2))
    expected = (inverse @ (direct.to_numpy() / output)) * output

    actual = two_sector_model.inoperability_loss(direct)

    np.testing.assert_allclose(actual.to_numpy(), expected)


@pytest.mark.parametrize("method_name", [
    "leontief_loss_from_final_demand",
    "ghosh_loss_from_value_added",
    "inoperability_loss",
])
def test_identity_model_returns_direct_loss(method_name: str) -> None:
    sectors = ["A", "B"]
    tables = TechnologyTables(
        direct_requirements=pd.DataFrame(
            np.zeros((2, 2)),
            index=sectors,
            columns=sectors,
        ),
        gross_output=pd.Series([100.0, 200.0], index=sectors),
        value_added=pd.DataFrame([[40.0, 80.0]], columns=sectors),
        final_demand=pd.DataFrame([[50.0], [100.0]], index=sectors),
    )
    model = InputOutputModel.from_tables(tables)
    direct = pd.Series({"A": 3.0, "B": 5.0})

    actual = getattr(model, method_name)(direct)

    np.testing.assert_allclose(actual.to_numpy(), direct.to_numpy())


def test_model_rejects_misaligned_sector_ids(two_sector_model: InputOutputModel) -> None:
    direct = pd.Series({"A": 3.0, "C": 5.0})

    with pytest.raises(ValueError, match="Missing.*B.*Extra.*C"):
        two_sector_model.leontief_loss_from_final_demand(direct)


def test_model_rejects_negative_loss(two_sector_model: InputOutputModel) -> None:
    direct = pd.Series({"A": -3.0, "B": 5.0})

    with pytest.raises(ValueError, match="nonnegative"):
        two_sector_model.ghosh_loss_from_value_added(direct)
