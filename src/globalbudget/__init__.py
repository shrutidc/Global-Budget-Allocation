"""globalbudget — reusable toolkit for the Global Budget Analysis project.

The package wraps loading, cleaning, feature engineering, visualization and
forecasting of the ``Master_Global_Budgets_Historical`` dataset (45 countries,
1936-2026, 9 government spending categories in both percentage and USD terms).

Typical usage::

    from globalbudget import data_loader, cleaning, features
    wide = data_loader.load_master()
    tidy = cleaning.to_long(wide)
"""

from . import cleaning, data_loader, features, forecasting, viz  # noqa: F401

# Canonical list of the nine spending categories (as they appear in column prefixes).
CATEGORIES = [
    "Defense",
    "Education",
    "Health",
    "Interest_Payments",
    "Infrastructure",
    "Agriculture",
    "State_Transfers",
    "Social_Welfare",
    "Administration_and_Others",
]

# Human-friendly labels for plots / reports.
CATEGORY_LABELS = {
    "Defense": "Defense",
    "Education": "Education",
    "Health": "Health",
    "Interest_Payments": "Interest Payments",
    "Infrastructure": "Infrastructure",
    "Agriculture": "Agriculture",
    "State_Transfers": "State Transfers",
    "Social_Welfare": "Social Welfare",
    "Administration_and_Others": "Administration & Others",
}

__all__ = [
    "cleaning",
    "data_loader",
    "features",
    "forecasting",
    "viz",
    "CATEGORIES",
    "CATEGORY_LABELS",
]

__version__ = "0.1.0"
