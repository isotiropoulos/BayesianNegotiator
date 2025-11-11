from __future__ import annotations

import re
from datetime import date

# Anchor for date normalization (used by utilities)
BASE_DATE = date(1, 1, 1)

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_ISO_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}"              # YYYY-MM-DD
    r"(?:[T ]"                         # T or space
    r"\d{2}:\d{2}:\d{2}"               # HH:MM:SS
    r"(?:\.\d+)?"                      # optional .milliseconds
    r"(?:Z|[+-]\d{2}:\d{2})?"          # Z or ±HH:MM
    r")?$"
)

# Full ISO-8601 duration: PnYnMnWnDTnHnMnS (any subset, order preserved)
# Examples: "P3Y6M4DT12H30M5S", "P2W", "PT45M", "P10D"
_ISO_DURATION_RE = re.compile(
    r"^P"
    r"(?:(?P<Y>\d+)Y)?"
    r"(?:(?P<Mo>\d+)M)?"
    r"(?:(?P<W>\d+)W)?"
    r"(?:(?P<D>\d+)D)?"
    r"(?:T"
    r"(?:(?P<H>\d+)H)?"
    r"(?:(?P<Mi>\d+)M)?"
    r"(?:(?P<S>\d+(?:\.\d+)?)S)?"
    r")?$"
)
