"""Parse Cell-ACDC/Trackastra output to measure daughter cell ID persistence.

Hypothesis: Trackastra produces stable daughter IDs across frames, so real splits
persist 10+ frames (confidence 1.0) while false positives persist 1 frame (confidence 0.1).

Compares against IoU tracker behavior where daughters lose IDs in 1-2 frames.
"""

from pathlib import Path
import pandas as pd
import numpy as np

ACDC_OUTPUT = Path(
    "G:/Projects/cell-split-counter-spike-cellacdc/data/cellacdc_input"
    "/Position_1/Images/video_acdc_output.csv"
)


def load_acdc_output(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
    return df


def find_division_events(df: pd.DataFrame) -> pd.DataFrame:
    """Find rows where a cell division is recorded."""
    div_cols = [c for c in df.columns if "division" in c.lower() or "relative_ID" in c.lower()]
    print(f"Division-related columns: {div_cols}")

    # Cell-ACDC marks divisions via 'division_frame_i' or 'relationship' column
    if "division_frame_i" in df.columns:
        divisions = df[df["division_frame_i"].notna() & (df["division_frame_i"] != 0)]
        print(f"\nFound {len(divisions)} division events via division_frame_i")
        return divisions

    if "relative_ID" in df.columns:
        # relative_ID > 0 means this cell is a daughter
        daughters = df[df["relative_ID"] > 0]
        print(f"\nFound {len(daughters)} daughter cell appearances via relative_ID")
        return daughters

    print("No division columns found — showing all columns:")
    print(df.columns.tolist())
    return pd.DataFrame()


def measure_daughter_persistence(df: pd.DataFrame) -> None:
    """For each daughter cell, measure how many frames it persists with its assigned ID."""
    if "relative_ID" not in df.columns:
        print("No relative_ID column — can't measure daughter persistence")
        return

    if "frame_i" not in df.columns:
        frame_col = [c for c in df.columns if "frame" in c.lower()]
        if not frame_col:
            print("No frame column found")
            return
        frame_col = frame_col[0]
    else:
        frame_col = "frame_i"

    if "Cell_ID" not in df.columns:
        id_col = [c for c in df.columns if "cell_id" in c.lower() or "id" == c.lower()]
        if not id_col:
            print("No Cell_ID column found")
            return
        id_col = id_col[0]
    else:
        id_col = "Cell_ID"

    # Find cells that were born as daughters (relative_ID > 0 at some frame)
    daughter_first_appearances = (
        df[df["relative_ID"] > 0]
        .groupby(id_col)[frame_col]
        .min()
        .rename("birth_frame")
    )

    persistence = []
    for cell_id, birth_frame in daughter_first_appearances.items():
        # Count consecutive frames this cell appears after birth
        cell_frames = sorted(df[df[id_col] == cell_id][frame_col].unique())
        consecutive = 0
        for f in cell_frames:
            if f >= birth_frame:
                if consecutive == 0 or f == birth_frame + consecutive:
                    consecutive += 1
                else:
                    break
        persistence.append({"cell_id": cell_id, "birth_frame": birth_frame, "frames_persisted": consecutive})

    pers_df = pd.DataFrame(persistence)
    if pers_df.empty:
        print("No daughter persistence data to show.")
        return

    print("\n=== Daughter Cell Persistence (Trackastra) ===")
    print(f"Total daughter cells: {len(pers_df)}")
    print(f"Median persistence: {pers_df['frames_persisted'].median():.1f} frames")
    print(f"Mean persistence: {pers_df['frames_persisted'].mean():.1f} frames")
    print(f"\nPersistence distribution:")
    bins = [1, 2, 3, 5, 10, 20, 50, 9999]
    labels = ["1", "2", "3", "4-5", "6-10", "11-20", "21-50", "51+"]
    pers_df["bin"] = pd.cut(pers_df["frames_persisted"], bins=[0]+bins, labels=["0"]+labels[:-1])
    print(pers_df["bin"].value_counts().sort_index())

    # Hypothesis check
    short_lived = (pers_df["frames_persisted"] <= 2).sum()
    long_lived = (pers_df["frames_persisted"] >= 10).sum()
    print(f"\nHypothesis check:")
    print(f"  Short-lived daughters (1-2 frames, likely false positives): {short_lived} ({100*short_lived/len(pers_df):.1f}%)")
    print(f"  Long-lived daughters (10+ frames, likely real splits): {long_lived} ({100*long_lived/len(pers_df):.1f}%)")

    # Compare to IoU tracker behavior: main pipeline detected ~210 splits but all short-lived
    print(f"\n  IoU tracker baseline: ~all daughters had 0-1 frame persistence (confidence ~0.1)")
    print(f"  If Trackastra shows bimodal distribution (1-2 vs 10+), hypothesis confirmed.")


def show_raw_summary(df: pd.DataFrame) -> None:
    print("\n=== Raw Data Summary ===")
    print(f"Frames: {df['frame_i'].min() if 'frame_i' in df.columns else '?'} to "
          f"{df['frame_i'].max() if 'frame_i' in df.columns else '?'}")
    print(f"Unique cells: {df['Cell_ID'].nunique() if 'Cell_ID' in df.columns else '?'}")
    print(f"\nFirst 5 rows:")
    print(df.head())


def main():
    if not ACDC_OUTPUT.exists():
        print(f"Output not yet written: {ACDC_OUTPUT}")
        print("Run this script after the Cell-ACDC pipeline completes.")
        return

    df = load_acdc_output(ACDC_OUTPUT)
    show_raw_summary(df)
    find_division_events(df)
    measure_daughter_persistence(df)


if __name__ == "__main__":
    main()
