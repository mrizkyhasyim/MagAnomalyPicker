import numpy as np
import pandas as pd


def nonlinear_despike(
    series: pd.Series, filter_width: int = 5, cutoff: float = 1.5
) -> pd.Series:
    """Applies a non-linear filter to remove high-frequency spikes exceeding cutoff."""
    # Rolling median background calculation
    med = series.rolling(
        window=filter_width, center=True, min_periods=1
    ).median()
    diff = (series - med).abs()

    cleaned = series.copy()
    cleaned[diff > cutoff] = np.nan
    return cleaned


def process_navigation(
    df: pd.DataFrame,
    x_col: str = "Easting",
    y_col: str = "Northing",
    cutoff: float = 1.5,
    max_gap: int = 50,
    lp_window: int = 100,
) -> pd.DataFrame:
    """Executes full navigation cleaning and distance QC pipeline."""
    # 1. De-spike
    df["Easting_DS"] = nonlinear_despike(
        df[x_col], filter_width=5, cutoff=cutoff
    )
    df["Northing_DS"] = nonlinear_despike(
        df[y_col], filter_width=5, cutoff=cutoff
    )

    # 2. Gap Interpolation & Rolling Low-Pass
    df["Easting_LP"] = (
        df["Easting_DS"]
        .interpolate(method="linear", limit=max_gap)
        .rolling(window=lp_window, center=True, min_periods=1)
        .mean()
    )
    df["Northing_LP"] = (
        df["Northing_DS"]
        .interpolate(method="linear", limit=max_gap)
        .rolling(window=lp_window, center=True, min_periods=1)
        .mean()
    )

    # 3. Track Distance QC Calculation
    dx = df["Easting_LP"].diff().fillna(0)
    dy = df["Northing_LP"].diff().fillna(0)
    df["DistQC"] = np.sqrt(dx**2 + dy**2)

    return df


def process_altitude(
    df: pd.DataFrame,
    alt_col: str = "Altitude",
    min_alt: float = 0.5,
    max_alt: float = 10.0,
    rs_window: int = 11,
) -> pd.DataFrame:
    """Executes altimeter windowing, gap filling, and rolling mean smoothing."""
    # 1. Windowing and De-spiking
    alt_ds = df[alt_col].copy()
    alt_ds[(alt_ds < min_alt) | (alt_ds > max_alt)] = np.nan
    df["Alt_DS"] = alt_ds

    # 2. Linear Gap Filling & Rolling Stat (Mean Filter)
    df["Alt_RS"] = (
        df["Alt_DS"]
        .interpolate(method="linear")
        .rolling(window=rs_window, center=True, min_periods=1)
        .mean()
    )

    return df