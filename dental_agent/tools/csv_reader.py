
import pandas as pd
from langchain_core.tools import tool

from dental_agent.config.settings import CSV_PATH, DATE_FORMAT


def clean_dataframe(df):
    """Replace NaN and missing values with JSON-safe None."""
    return df.astype(object).where(pd.notna(df), None)


def format_dates(df):
    """Format date_slot consistently across Windows and Linux."""
    df = df.copy()
    df["date_slot"] = df["date_slot"].apply(
        lambda dt: (
            f"{dt.month}/{dt.day}/{dt.year} "
            f"{dt.hour}:{dt.minute:02d}"
        )
    )
    return df


def _load_df() -> pd.DataFrame:
    df = pd.read_csv(
        CSV_PATH,
        dtype={"patient_to_attend": "string"}
    )

    df.columns = df.columns.str.strip()

    df["is_available"] = (
        df["is_available"].astype(str).str.upper() == "TRUE"
    )

    df["date_slot"] = pd.to_datetime(
        df["date_slot"],
        format="mixed",
        dayfirst=False
    )

    df["doctor_name"] = (
        df["doctor_name"].str.lower().str.strip()
    )

    df["specialization"] = (
        df["specialization"].str.lower().str.strip()
    )

    # Normalize patient IDs and remove missing values
    df["patient_to_attend"] = (
        df["patient_to_attend"]
        .fillna("")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    return df


@tool
def get_available_slots(
    specialization: str = "",
    doctor_name: str = "",
    date_filter: str = "",
) -> list:
    """Return available appointment slots."""

    df = _load_df()
    mask = df["is_available"]

    if specialization:
        mask &= (
            df["specialization"] == specialization.lower().strip()
        )

    if doctor_name:
        mask &= (
            df["doctor_name"] == doctor_name.lower().strip()
        )

    if date_filter:
        try:
            target_date = pd.to_datetime(date_filter).date()
            mask &= df["date_slot"].dt.date == target_date
        except Exception:
            pass

    result = df.loc[
        mask,
        ["date_slot", "specialization", "doctor_name"]
    ].copy()

    result = format_dates(result)
    result = clean_dataframe(result)

    return result.head(20).to_dict(orient="records")


@tool
def get_patient_appointments(patient_id: str) -> list:
    """Return all booked appointments for a patient."""

    df = _load_df()

    mask = (
        df["patient_to_attend"] == str(patient_id).strip()
    )

    result = df.loc[
        mask,
        [
            "date_slot",
            "specialization",
            "doctor_name",
            "patient_to_attend"
        ]
    ].copy()

    result = format_dates(result)
    result = clean_dataframe(result)

    return result.to_dict(orient="records")


@tool
def check_slot_availability(
    doctor_name: str,
    date_slot: str
) -> dict:
    """Check whether a specific doctor's slot is available."""

    df = _load_df()

    try:
        target_dt = pd.to_datetime(
            date_slot,
            format="mixed",
            dayfirst=False
        )
    except Exception:
        return {
            "found": False,
            "is_available": False,
            "patient_to_attend": ""
        }

    mask = (
        (df["doctor_name"] == doctor_name.lower().strip())
        & (df["date_slot"] == target_dt)
    )

    rows = df[mask]

    if rows.empty:
        return {
            "found": False,
            "is_available": False,
            "patient_to_attend": ""
        }

    row = rows.iloc[0]

    return {
        "found": True,
        "is_available": bool(row["is_available"]),
        "patient_to_attend": str(row["patient_to_attend"] or "")
    }


@tool
def list_doctors_by_specialization(
    specialization: str
) -> list:
    """Return distinct doctors for a specialization."""

    df = _load_df()

    mask = (
        df["specialization"] == specialization.lower().strip()
    )

    return sorted(
        df.loc[mask, "doctor_name"].dropna().unique().tolist()
    )
