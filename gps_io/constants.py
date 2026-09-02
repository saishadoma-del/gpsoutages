# Role: Canonical metadata for the GPS input-output analysis
# Author: Saisha Doma
# Description: Defines sector IDs, aggregation codes, dependency ranges, and durations.

from typing import Final

SECTOR_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "AGR": ("11",),
    "MINING": ("21",),
    "UTILITIES": ("22",),
    "CONSTR": ("23",),
    "MANUF": ("31G",),
    "TRADE": ("42", "44RT"),
    "TRANS": ("48TW",),
    "INFO": ("51",),
    "FIRE": ("FIRE",),
    "PROF_OTHER": ("PROF", "81"),
    "EDUC_HEALTH": ("6",),
    "ART_ENT": ("7",),
    "G": ("G",),
}

SECTOR_IDS: Final[tuple[str, ...]] = tuple(SECTOR_GROUPS)

SECTOR_LABELS: Final[dict[str, str]] = {
    "AGR": "Agriculture",
    "MINING": "Mining and Oil/Gas",
    "UTILITIES": "Utilities",
    "CONSTR": "Construction",
    "MANUF": "Manufacturing",
    "TRADE": "Trade",
    "TRANS": "Transportation",
    "INFO": "Information",
    "FIRE": "Finance and Real Estate",
    "PROF_OTHER": "Professional Services",
    "EDUC_HEALTH": "Education and Health",
    "ART_ENT": "Arts and Entertainment",
    "G": "Government",
}

GPS_DEPENDENCY_RANGES: Final[dict[str, tuple[float, float]]] = {
    "AGR": (0.00, 0.10),
    "MINING": (0.00, 0.10),
    "UTILITIES": (0.30, 0.40),
    "CONSTR": (0.10, 0.20),
    "MANUF": (0.10, 0.20),
    "TRADE": (0.00, 0.10),
    "TRANS": (0.30, 0.40),
    "INFO": (0.20, 0.30),
    "FIRE": (0.10, 0.20),
    "PROF_OTHER": (0.10, 0.20),
    "EDUC_HEALTH": (0.00, 0.10),
    "ART_ENT": (0.00, 0.10),
    "G": (0.00, 0.10),
}

OUTAGE_HOURS: Final[tuple[int, ...]] = (1, 6, 12, 24)
HOURS_PER_YEAR: Final[int] = 365 * 24

VALUE_ADDED_ROWS: Final[tuple[str, ...]] = (
    "V001",
    "V003",
    "T00OTOP",
    "T00OSUB",
)

FINAL_DEMAND_COLUMNS: Final[tuple[str, ...]] = (
    "F010",
    "F100",
    "F020",
    "F030",
    "F040",
)

TOTAL_INDUSTRY_OUTPUT_ROW: Final[str] = "T018"
