from __future__ import annotations
import matplotlib.pyplot as plt

from pathlib import Path

import pandas as pd
import seaborn as sns

import matplotlib
from matplotlib.container import BarContainer

# to directly save plots to the fs
matplotlib.use("Agg")


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "analysis_outputs"

# specs to clean both datasets
NA_VALUES = ["NA", "N/A", "na", "none", "None", "null", "NULL", ""]
NUMERIC_MATCH_COLUMNS = ["id", "target_runs", "target_overs", "result_margin"]
NUMERIC_DELIVERY_COLUMNS = [
    "match_id",
    "inning",
    "over",
    "ball",
    "batsman_runs",
    "extra_runs",
    "total_runs",
    "is_wicket",
]

# sns style guide
sns.set_theme(style="whitegrid", context="talk")


def _clean_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and normalize common missing-value markers."""
    cleaned = df.copy()
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].astype("string").str.strip()
        cleaned[column] = cleaned[column].replace(NA_VALUES, pd.NA)
    return cleaned


def load_and_clean_matches(path: str) -> pd.DataFrame:
    matches = pd.read_csv(path, na_values=NA_VALUES, keep_default_na=True)
    matches = matches.drop_duplicates().copy()
    matches = _clean_object_columns(matches)

    for column in NUMERIC_MATCH_COLUMNS:
        if column in matches.columns:
            matches[column] = pd.to_numeric(matches[column], errors="coerce")

    if "date" in matches.columns:
        matches["date"] = pd.to_datetime(matches["date"], errors="coerce")

    if "season" in matches.columns:
        matches["season"] = matches["season"].astype("string").str.strip()

    return matches


def load_and_clean_deliveries(path: str) -> pd.DataFrame:
    deliveries = pd.read_csv(path, na_values=NA_VALUES, keep_default_na=True)
    deliveries = deliveries.drop_duplicates().copy()
    deliveries = _clean_object_columns(deliveries)

    for column in NUMERIC_DELIVERY_COLUMNS:
        if column in deliveries.columns:
            deliveries[column] = pd.to_numeric(deliveries[column], errors="coerce")

    for column in ["over", "ball", "inning", "match_id", "is_wicket"]:
        if column in deliveries.columns:
            deliveries[column] = deliveries[column].fillna(0).astype(int)

    for column in ["batsman_runs", "extra_runs", "total_runs"]:
        if column in deliveries.columns:
            deliveries[column] = deliveries[column].fillna(0).astype(int)

    return deliveries


def build_wicket_analysis(
    matches: pd.DataFrame, deliveries: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    required_match_columns = {"id", "season", "date"}
    required_delivery_columns = {"match_id", "over", "is_wicket"}

    missing_match = required_match_columns.difference(matches.columns)
    missing_delivery = required_delivery_columns.difference(deliveries.columns)

    if missing_match:
        raise ValueError(f"matches.csv is missing required columns: {
                         sorted(missing_match)}")
    if missing_delivery:
        raise ValueError(f"deliveries.csv is missing required columns: {
                         sorted(missing_delivery)}")

    # constructing a merged dataset to make analysis easier
    merged = deliveries.merge(
        matches[["id", "season", "date"]],
        left_on="match_id",
        right_on="id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_match"),
    )

    missing_season_rows = int(merged["season"].isna().sum())
    if missing_season_rows:
        raise ValueError(
            f"{missing_season_rows} deliveries could not be mapped to a season."
        )

    wicket_events = merged.loc[merged["is_wicket"].eq(1)].copy()
    wicket_events["over_number"] = wicket_events["over"] + 1

    max_over = int(wicket_events["over_number"].max())
    over_axis = list(range(1, max_over + 1))

    overall = (
        wicket_events.groupby("over_number", as_index=False)
        .size()
        .rename(columns={"size": "wickets"})
        .set_index("over_number")
        .reindex(over_axis, fill_value=0)
        .reset_index()
    )

    season_order = (
        matches.dropna(subset=["season", "date"])
        .sort_values(["date", "id"])
        .drop_duplicates(subset=["season"])["season"]
        .tolist()
    )
    if len(season_order) == 0:
        season_order = sorted(wicket_events["season"].dropna().unique().tolist())

    season_over = (
        wicket_events.groupby(["season", "over_number"], as_index=False)
        .size()
        .rename(columns={"size": "wickets"})
    )
    season_matrix = (
        season_over.pivot(index="season", columns="over_number", values="wickets")
        .reindex(index=season_order, columns=over_axis)
        .fillna(0)
        .astype(int)
    )

    top_over = overall.loc[overall["wickets"].idxmax()]
    top_over_number = top_over["over_number"].astype(int)
    top_over_wickets = top_over["wickets"].astype(int)
    summary_lines = [
        f"Total matches loaded: {len(matches):,}",
        f"Total deliveries loaded: {len(deliveries):,}",
        f"Total wicket events: {len(wicket_events):,}",
        f"Top wicket over overall: Over {
            top_over_number} with {top_over_wickets} wickets",
    ]

    return wicket_events, overall, season_matrix, summary_lines


def save_overall_plot(overall: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=overall, x="over_number", y="wickets", ax=ax, color="#5B8FF9")
    ax.set_title("Wickets by Over Across the Full IPL Dataset")
    ax.set_xlabel("Over number")
    ax.set_ylabel("Wickets")
    ax.set_xticks(range(len(overall)))
    ax.set_xticklabels(overall["over_number"])

    for container in ax.containers:
        if isinstance(container, BarContainer):
            ax.bar_label(container, fmt="%.0f", padding=2, fontsize=9)

    fig.tight_layout()
    output_path = output_dir / "wickets_by_over_overall.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_season_heatmap(season_matrix: pd.DataFrame, output_dir: Path) -> Path:
    height = max(6, 0.45 * len(season_matrix))
    fig, ax = plt.subplots(figsize=(16, height))
    sns.heatmap(
        season_matrix,
        ax=ax,
        cmap="YlOrRd",
        annot=True,
        fmt="d",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Wickets"},
    )
    ax.set_title("Wickets by Over and Season")
    ax.set_xlabel("Over number")
    ax.set_ylabel("Season")
    fig.tight_layout()
    output_path = output_dir / "wickets_by_over_season.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_sanity_checks(
    wicket_events: pd.DataFrame, overall: pd.DataFrame, season_matrix: pd.DataFrame
) -> None:
    total_wickets = int(wicket_events.shape[0])
    overall_total = int(overall["wickets"].sum())
    season_total = int(season_matrix.to_numpy().sum())
    missing_seasons = int(wicket_events["season"].isna().sum())

    if overall_total != total_wickets:
        raise AssertionError(
            "Overall wicket counts do not match the wicket events table."
        )
    if season_total != total_wickets:
        raise AssertionError(
            "Season-wise wicket counts do not match the wicket events table."
        )
    if missing_seasons != 0:
        raise AssertionError("Some wicket events are missing season information.")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    matches = load_and_clean_matches("assets/matches.csv")
    deliveries = load_and_clean_deliveries("assets/deliveries.csv")

    wicket_events, overall, season_matrix, summary_lines = build_wicket_analysis(
        matches, deliveries
    )
    run_sanity_checks(wicket_events, overall, season_matrix)

    overall_path = save_overall_plot(overall, OUTPUT_DIR)
    season_path = save_season_heatmap(season_matrix, OUTPUT_DIR)

    overall_csv = OUTPUT_DIR / "wickets_by_over_overall.csv"
    season_csv = OUTPUT_DIR / "wickets_by_over_season.csv"
    overall.to_csv(overall_csv, index=False)
    season_matrix.to_csv(season_csv)

    print("\n".join(summary_lines))
    print()
    print("Wickets by over (overall):")
    print(overall.to_string(index=False))
    print()
    print("Season-wise wicket over matrix:")
    print(season_matrix.to_string())
    print()
    print("Saved plots to:")
    print("- " + str(overall_path))
    print("- " + str(season_path))
    print()
    print("Saved summary tables to:")
    print("- " + str(overall_csv))
    print("- " + str(season_csv))


if __name__ == "__main__":
    main()
