from __future__ import annotations

import io
import sqlite3
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------

URL_NORWAY = "https://www.football-data.co.uk/new/NOR.csv"
START_YEAR = 2023

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "eliteserien.db"
RAW_CSV_PATH = DATA_DIR / "NOR_raw.csv"
FILTERED_CSV_PATH = DATA_DIR / "eliteserien_2023_onwards.csv"


# ---------------------------------------------------------
# UTILIDADES
# ---------------------------------------------------------

def pick_col(df: pd.DataFrame, *names: str) -> str | None:
    """Devuelve la primera columna existente entre varias alternativas."""
    for name in names:
        if name in df.columns:
            return name
    return None


def series_or_none(df: pd.DataFrame, *names: str) -> pd.Series:
    """Devuelve una Serie existente o una Serie de None si no existe."""
    col = pick_col(df, *names)
    if col is None:
        return pd.Series([None] * len(df), index=df.index)
    return df[col]


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def download_csv() -> bytes:
    print("Descargando datos gratuitos de Football-Data.co.uk...")
    print(URL_NORWAY)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/120 Safari/537.36"
        )
    }

    response = requests.get(
        URL_NORWAY,
        headers=headers,
        timeout=45,
    )
    response.raise_for_status()

    if not response.content:
        raise RuntimeError("La descarga llegó vacía.")

    return response.content


def load_dataframe(content: bytes) -> pd.DataFrame:
    # utf-8-sig suele funcionar; latin1 queda como respaldo.
    last_error = None

    for encoding in ("utf-8-sig", "latin1"):
        try:
            df = pd.read_csv(
                io.BytesIO(content),
                encoding=encoding,
                low_memory=False,
            )
            if len(df.columns) > 1:
                break
        except Exception as exc:
            last_error = exc
    else:
        raise RuntimeError(
            f"No se pudo leer el CSV descargado. Último error: {last_error}"
        )

    df.columns = [str(c).strip() for c in df.columns]

    # Elimina filas completamente vacías.
    df = df.dropna(how="all").copy()

    return df


def parse_dates(df: pd.DataFrame) -> pd.Series:
    date_col = pick_col(df, "Date", "date", "DATE")

    if date_col is None:
        raise RuntimeError(
            "El fichero no contiene una columna de fecha reconocible."
        )

    # Noruega usa normalmente formato día/mes/año.
    parsed = pd.to_datetime(
        df[date_col],
        dayfirst=True,
        errors="coerce",
    )

    return parsed


def filter_from_2023(df: pd.DataFrame) -> pd.DataFrame:
    parsed_dates = parse_dates(df)
    filtered = df.loc[parsed_dates.dt.year >= START_YEAR].copy()
    filtered["parsed_date"] = parsed_dates.loc[filtered.index]

    # Noruega juega por año natural, así que si no existe Season,
    # la temporada se puede derivar de la fecha.
    if "Season" not in filtered.columns:
        filtered["Season"] = filtered["parsed_date"].dt.year

    return filtered.sort_values("parsed_date").reset_index(drop=True)


def build_normalized_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea una tabla estable para nuestra app.

    Si Football-Data incluye una estadística, la guardamos.
    Si no la incluye, queda vacía y posteriormente se completará
    con la fuente estadística detallada.
    """

    out = pd.DataFrame(index=df.index)

    out["season"] = clean_numeric(series_or_none(df, "Season"))
    out["date"] = df["parsed_date"].dt.strftime("%Y-%m-%d")

    time_col = pick_col(df, "Time", "time")
    out["time"] = (
        df[time_col].astype(str)
        if time_col
        else None
    )

    out["league"] = series_or_none(df, "League", "Div")
    out["home_team"] = series_or_none(df, "Home", "HomeTeam")
    out["away_team"] = series_or_none(df, "Away", "AwayTeam")

    out["home_goals"] = clean_numeric(
        series_or_none(df, "HG", "FTHG")
    )
    out["away_goals"] = clean_numeric(
        series_or_none(df, "AG", "FTAG")
    )
    out["result"] = series_or_none(df, "Res", "FTR")

    # Estadísticas detalladas: se guardan automáticamente
    # si aparecen en la fuente.
    out["home_shots"] = clean_numeric(series_or_none(df, "HS"))
    out["away_shots"] = clean_numeric(series_or_none(df, "AS"))
    out["home_shots_on_target"] = clean_numeric(series_or_none(df, "HST"))
    out["away_shots_on_target"] = clean_numeric(series_or_none(df, "AST"))
    out["home_corners"] = clean_numeric(series_or_none(df, "HC"))
    out["away_corners"] = clean_numeric(series_or_none(df, "AC"))
    out["home_fouls"] = clean_numeric(series_or_none(df, "HF"))
    out["away_fouls"] = clean_numeric(series_or_none(df, "AF"))
    out["home_yellow"] = clean_numeric(series_or_none(df, "HY"))
    out["away_yellow"] = clean_numeric(series_or_none(df, "AY"))
    out["home_red"] = clean_numeric(series_or_none(df, "HR"))
    out["away_red"] = clean_numeric(series_or_none(df, "AR"))
    out["referee"] = series_or_none(df, "Referee")

    # Algunas cuotas comunes de Football-Data.
    for source_col, target_col in [
        ("PH", "odds_home"),
        ("PD", "odds_draw"),
        ("PA", "odds_away"),
        ("AvgH", "odds_home_avg"),
        ("AvgD", "odds_draw_avg"),
        ("AvgA", "odds_away_avg"),
        ("MaxH", "odds_home_max"),
        ("MaxD", "odds_draw_max"),
        ("MaxA", "odds_away_max"),
        ("PSCH", "odds_home_close"),
        ("PSCD", "odds_draw_close"),
        ("PSCA", "odds_away_close"),
    ]:
        out[target_col] = clean_numeric(
            series_or_none(df, source_col)
        )

    return out


def save_database(raw_df: pd.DataFrame, normalized_df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        raw_df.to_sql(
            "football_data_raw",
            conn,
            if_exists="replace",
            index=False,
        )

        normalized_df.to_sql(
            "matches",
            conn,
            if_exists="replace",
            index=False,
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_matches_date
            ON matches(date)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_matches_home
            ON matches(home_team)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_matches_away
            ON matches(away_team)
            """
        )


def print_summary(source_df: pd.DataFrame, filtered_df: pd.DataFrame, normalized_df: pd.DataFrame) -> None:
    print()
    print("=" * 65)
    print("ELITESERIEN EDGE PRO - ACTUALIZACION COMPLETADA")
    print("=" * 65)

    print(f"Filas totales descargadas: {len(source_df)}")
    print(f"Partidos desde {START_YEAR}: {len(filtered_df)}")

    seasons = (
        normalized_df["season"]
        .dropna()
        .astype(int)
        .sort_values()
        .unique()
        .tolist()
    )
    print(f"Temporadas guardadas: {seasons}")

    teams = sorted(
        set(normalized_df["home_team"].dropna().astype(str))
        | set(normalized_df["away_team"].dropna().astype(str))
    )
    print(f"Equipos distintos: {len(teams)}")

    print()
    print("Columnas que trae actualmente el CSV de Noruega:")
    print(", ".join(map(str, source_df.columns.tolist())))

    print()
    print("Cobertura detectada en nuestra tabla:")

    checks = [
        ("Goles", ["home_goals", "away_goals"]),
        ("Tiros", ["home_shots", "away_shots"]),
        ("Tiros a puerta", ["home_shots_on_target", "away_shots_on_target"]),
        ("Corners", ["home_corners", "away_corners"]),
        ("Faltas", ["home_fouls", "away_fouls"]),
        ("Amarillas", ["home_yellow", "away_yellow"]),
        ("Rojas", ["home_red", "away_red"]),
        ("Arbitro", ["referee"]),
    ]

    for label, cols in checks:
        available = any(
            normalized_df[col].notna().any()
            for col in cols
            if col in normalized_df.columns
        )
        print(f"  {'OK' if available else '--'} {label}")

    print()
    print(f"Base SQLite: {DB_PATH}")
    print(f"CSV filtrado: {FILTERED_CSV_PATH}")
    print("=" * 65)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        content = download_csv()

        RAW_CSV_PATH.write_bytes(content)

        source_df = load_dataframe(content)

        filtered_df = filter_from_2023(source_df)
        filtered_df.to_csv(
            FILTERED_CSV_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        normalized_df = build_normalized_table(filtered_df)

        # Quitamos filas sin equipos por seguridad.
        normalized_df = normalized_df[
            normalized_df["home_team"].notna()
            & normalized_df["away_team"].notna()
        ].reset_index(drop=True)

        save_database(filtered_df, normalized_df)
        print_summary(source_df, filtered_df, normalized_df)

    except requests.RequestException as exc:
        print()
        print("ERROR descargando Football-Data:")
        print(exc)
        raise SystemExit(1)

    except Exception as exc:
        print()
        print("ERROR durante la actualización:")
        print(exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
