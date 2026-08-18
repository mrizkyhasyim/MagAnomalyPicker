from pathlib import Path
import numpy as np
import pandas as pd

STANDARD_MAG_HEADERS = [
    "Date",
    "Time",
    "Line_Name",
    "Easting",
    "Northing",
    "Towfish",
    "SOG",
    "STW",
    "Altitude_Raw",
    "Depth_Raw",
    "Layback_Raw",
    "Total_Field_Raw",
    "Signal_Strength_Raw",
]


def process_time_qc(
    df: pd.DataFrame, time_col: str = "Time", expected_dt: float = 0.1
) -> pd.DataFrame:
    if time_col not in df.columns:
        return df
    try:
        td = pd.to_timedelta(df[time_col].astype(str))
        df["Time_Sec"] = td.dt.total_seconds().values

        time_sec = df["Time_Sec"].values
        dt = np.zeros_like(time_sec)
        dt[1:] = np.diff(time_sec)
        dt[0] = 0.0
        df["Time_Diff"] = dt

        qc_mask = np.zeros(len(df), dtype=int)
        qc_mask[dt > (expected_dt * 1.5)] = 1
        qc_mask[dt == 0] = 2
        qc_mask[dt < 0] = 3
        qc_mask[0] = 0
        df["Time_QC_Mask"] = qc_mask
    except Exception as e:
        print(f"Time QC warning: {e}")
    return df


def load_dataset(
    file_path: str | Path,
    delimiter: str | None = "Auto",
    has_header: str | None = "Auto-detect",
    comment_char: str | None = "None",
    drop_col_indices: list[int] | None = None,
    new_header_names: list[str] | None = None,
    rename_dict: dict[str, str] | None = None,
    dummy_vals: list = [-99999, -9999, "*", "*.*", "1.#QNAN", "1.#IND"],
    expected_dt: float = 0.1,  # Default 10 Hz sampling rate
) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    first_line = ""
    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith(("#", "//", "!")):
                first_line = stripped
                break

    read_kwargs = {}
    if delimiter in [None, "Auto"]:
        if "," in first_line:
            sep = ","
            read_kwargs["skipinitialspace"] = True
        elif "\t" in first_line:
            sep = "\t"
        elif ";" in first_line:
            sep = ";"
        else:
            sep = r"\s+"
            read_kwargs["engine"] = "python"
    elif delimiter == "Comma":
        sep = ","
        read_kwargs["skipinitialspace"] = True
    elif delimiter == "Tab":
        sep = "\t"
    elif delimiter == "Semicolon":
        sep = ";"
    elif delimiter == "Colon":
        sep = ":"
    elif delimiter == "Whitespace":
        sep = r"\s+"
        read_kwargs["engine"] = "python"
    else:
        sep = delimiter

    read_kwargs["sep"] = sep

    is_headerless = False
    if has_header == "No Header":
        is_headerless = True
    elif has_header == "Has Header":
        is_headerless = False
    else:
        first_token = (
            first_line.split(",")[0].split()[0]
            if "," in first_line
            else first_line.split()[0]
        )
        clean_token = (
            first_token.replace("/", "").replace("-", "").replace(".", "")
        )
        if clean_token.isdigit():
            is_headerless = True

    if is_headerless:
        read_kwargs["header"] = None

    if comment_char and comment_char != "None":
        read_kwargs["comment"] = comment_char

    df = pd.read_csv(path, **read_kwargs)

    if is_headerless:
        if len(df.columns) == len(STANDARD_MAG_HEADERS):
            df.columns = STANDARD_MAG_HEADERS
        else:
            df.columns = [f"Col_{i+1}" for i in range(len(df.columns))]

    df.columns = df.columns.astype(str).str.strip()
    df.replace(dummy_vals, np.nan, inplace=True)

    if drop_col_indices:
        valid_indices = [
            i for i in drop_col_indices if 0 <= i < len(df.columns)
        ]
        if valid_indices:
            df.drop(columns=df.columns[valid_indices], inplace=True)

    if new_header_names and len(new_header_names) == len(df.columns):
        df.columns = [h.strip() for h in new_header_names]

    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)

    # Force numeric type across numeric channels
    text_metadata_cols = ["Date", "Time", "Line_Name"]
    for col in df.columns:
        if col not in text_metadata_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Execute Time QC Processing
    df = process_time_qc(df, time_col="Time", expected_dt=expected_dt)

    if "Mask" not in df.columns:
        df["Mask"] = 1

    if "Easting" in df.columns:
        df["Easting_Edits"] = df["Easting"].copy()
    if "Northing" in df.columns:
        df["Northing_Edits"] = df["Northing"].copy()

    return df