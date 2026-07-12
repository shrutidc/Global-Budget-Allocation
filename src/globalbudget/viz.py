"""Reusable plotting helpers with a consistent theme and category palette."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .data_loader import CATEGORIES, PCT_COLS, PROJECT_ROOT

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# A stable, colour-blind-friendly palette mapped to the nine categories.
_PALETTE = sns.color_palette("colorblind", n_colors=len(CATEGORIES))
CATEGORY_COLORS = dict(zip(CATEGORIES, _PALETTE))

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


def set_theme() -> None:
    """Apply the project-wide matplotlib/seaborn theme. Call once per notebook."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.figsize": (11, 6),
            "figure.dpi": 110,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "savefig.bbox": "tight",
            "savefig.dpi": 150,
        }
    )


def save(fig, name: str) -> Path:
    """Save a figure to ``reports/figures/<name>.png`` and return the path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    return path


def plot_total_budget(df: pd.DataFrame, countries: list[str], ax=None):
    """Line plot of total budget over time for the given countries."""
    if ax is None:
        _, ax = plt.subplots()
    for country in countries:
        sub = df[df["Country"] == country].sort_values("Year")
        ax.plot(sub["Year"], sub["Total_Budget_Billions_USD"], label=country, lw=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Total budget (USD billions)")
    ax.set_title("Total government budget over time")
    ax.legend(ncol=2, fontsize=9)
    return ax


def plot_composition(df: pd.DataFrame, country: str, ax=None):
    """Stacked-area chart of budget composition (%) over time for one country."""
    if ax is None:
        _, ax = plt.subplots()
    sub = df[df["Country"] == country].sort_values("Year")
    ax.stackplot(
        sub["Year"],
        *[sub[f"{c}_Percentage"] for c in CATEGORIES],
        labels=[CATEGORY_LABELS[c] for c in CATEGORIES],
        colors=[CATEGORY_COLORS[c] for c in CATEGORIES],
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of budget (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Budget composition over time — {country}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    return ax


def plot_category_heatmap(df: pd.DataFrame, category: str, ax=None):
    """Heatmap of a single category's share (%) across countries and years."""
    col = f"{category}_Percentage"
    if col not in df.columns:
        raise ValueError(f"Unknown category '{category}'. Choose from {CATEGORIES}.")
    pivot = df.pivot_table(index="Country", columns="Year", values=col)
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(pivot, cmap="rocket_r", ax=ax, cbar_kws={"label": f"{category} (%)"})
    ax.set_title(f"{CATEGORY_LABELS.get(category, category)} share of budget (%)")
    return ax
