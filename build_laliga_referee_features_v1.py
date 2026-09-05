from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "laliga.db"

SOURCE_TABLE = "referee_match_history"
OUTPUT_TABLE = "referee_pre_match_features"
CURRENT_TABLE = "referee_current_summary_v2"

# Cuántos partidos equivalentes damos al prior de liga para suavizar árbitros
# con muestras pequeñas.
PRIOR_STRENGTH = 15


def sql_table_exists(conn, table_name):
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type='table' AND name=?
        LIMIT 1
        """,
        (table_name,),
    ).fetchone()
    return row is not None


def safe_div(num, den):
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    return num / den.replace(0, np.nan)


def rolling_prior(series, window):
    return (
        pd.to_numeric(series, errors="coerce")
        .shift(1)
        .rolling(window=window, min_periods=1)
        .mean()
    )


def expanding_prior(series):
    return (
        pd.to_numeric(series, errors="coerce")
        .shift(1)
        .expanding(min_periods=1)
        .mean()
    )


def referee_group_features(group):
    g = group.sort_values(["utc_time", "match_id"]).copy()

    g["ref_matches_before"] = np.arange(len(g), dtype=int)

    for col, prefix in [
        ("total_yellow", "yellow"),
        ("total_red", "red"),
        ("total_fouls", "fouls"),
    ]:
        s = pd.to_numeric(g[col], errors="coerce")

        g[f"ref_{prefix}_avg_before"] = expanding_prior(s)
        g[f"ref_{prefix}_last5_before"] = rolling_prior(s, 5)
        g[f"ref_{prefix}_last10_before"] = rolling_prior(s, 10)
        g[f"ref_{prefix}_last20_before"] = rolling_prior(s, 20)

    return g


def profile_from_delta(delta):
    if pd.isna(delta):
        return "SIN MUESTRA"
    if delta >= 0.60:
        return "MUY TARJETERO"
    if delta >= 0.25:
        return "TARJETERO"
    if delta <= -0.60:
        return "MUY PERMISIVO"
    if delta <= -0.25:
        return "PERMISIVO"
    return "NEUTRO"


def build_current_summary(history):
    valid = history.dropna(subset=["referee"]).copy()
    valid = valid.sort_values(["utc_time", "match_id"])

    league_avg_y = pd.to_numeric(
        valid["total_yellow"], errors="coerce"
    ).mean()
    league_avg_f = pd.to_numeric(
        valid["total_fouls"], errors="coerce"
    ).mean()
    league_avg_r = pd.to_numeric(
        valid["total_red"], errors="coerce"
    ).mean()

    rows = []

    for referee, g in valid.groupby("referee", sort=False):
        g = g.sort_values(["utc_time", "match_id"])
        y = pd.to_numeric(g["total_yellow"], errors="coerce")
        f = pd.to_numeric(g["total_fouls"], errors="coerce")
        r = pd.to_numeric(g["total_red"], errors="coerce")

        n_y = int(y.notna().sum())
        raw_y = float(y.mean()) if n_y else np.nan

        shrunk_y = (
            (n_y * raw_y + PRIOR_STRENGTH * league_avg_y)
            / (n_y + PRIOR_STRENGTH)
            if n_y
            else league_avg_y
        )

        delta = shrunk_y - league_avg_y

        rows.append({
            "referee": referee,
            "matches": int(len(g)),
            "matches_with_yellow": n_y,
            "avg_yellow": raw_y,
            "last5_yellow": float(y.tail(5).mean()),
            "last10_yellow": float(y.tail(10).mean()),
            "last20_yellow": float(y.tail(20).mean()),
            "shrunk_yellow": float(shrunk_y),
            "league_avg_yellow": float(league_avg_y),
            "yellow_delta_vs_league": float(delta),
            "avg_fouls": float(f.mean()),
            "last10_fouls": float(f.tail(10).mean()),
            "avg_red": float(r.mean()),
            "first_match": g["utc_time"].min().isoformat()
                if pd.notna(g["utc_time"].min()) else None,
            "last_match": g["utc_time"].max().isoformat()
                if pd.notna(g["utc_time"].max()) else None,
            "profile": profile_from_delta(delta),
        })

    return pd.DataFrame(rows).sort_values(
        ["matches", "shrunk_yellow"],
        ascending=[False, False],
    )


def main():
    print()
    print("=" * 92)
    print("LALIGA EDGE PRO - FEATURES PRE-PARTIDO DE ARBITROS")
    print("=" * 92)

    if not DB_PATH.exists():
        raise SystemExit(f"No existe la base de datos: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        if not sql_table_exists(conn, SOURCE_TABLE):
            raise SystemExit(
                f"No existe la tabla {SOURCE_TABLE}. "
                "Primero termina la descarga de árbitros."
            )

        df = pd.read_sql_query(
            f"SELECT * FROM {SOURCE_TABLE}",
            conn,
        )

    required = {
        "match_id",
        "season",
        "utc_time",
        "home_team",
        "away_team",
        "referee",
        "total_yellow",
        "total_red",
        "total_fouls",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(
            "Faltan columnas en referee_match_history: "
            + ", ".join(missing)
        )

    print(f"Filas originales: {len(df)}")

    dupes = int(df["match_id"].duplicated(keep=False).sum())
    print(f"Filas pertenecientes a match_id duplicados: {dupes}")

    # Si hubiera duplicados accidentales conservamos una sola fila por partido.
    df = (
        df.sort_values(["season", "utc_time", "match_id"])
        .drop_duplicates(subset=["match_id"], keep="last")
        .copy()
    )

    df["utc_time"] = pd.to_datetime(
        df["utc_time"],
        utc=True,
        errors="coerce",
    )

    df["referee"] = (
        df["referee"]
        .astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    print(f"Partidos únicos: {len(df)}")
    print(f"Sin fecha válida: {int(df['utc_time'].isna().sum())}")
    print(f"Sin árbitro: {int(df['referee'].isna().sum())}")
    print(
        "Sin total de amarillas: "
        f"{int(pd.to_numeric(df['total_yellow'], errors='coerce').isna().sum())}"
    )
    print(
        "Sin total de faltas: "
        f"{int(pd.to_numeric(df['total_fouls'], errors='coerce').isna().sum())}"
    )

    missing_card_rows = df[
        pd.to_numeric(df["total_yellow"], errors="coerce").isna()
    ][
        ["match_id", "season", "utc_time", "home_team", "away_team", "referee"]
    ]

    if not missing_card_rows.empty:
        print()
        print("PARTIDOS SIN AMARILLAS (se excluirán del target de tarjetas):")
        print(missing_card_rows.to_string(index=False))

    # Orden cronológico global para construir el prior de liga SIN futuro.
    df = df.sort_values(["utc_time", "match_id"]).reset_index(drop=True)

    yellow = pd.to_numeric(df["total_yellow"], errors="coerce")
    fouls = pd.to_numeric(df["total_fouls"], errors="coerce")
    red = pd.to_numeric(df["total_red"], errors="coerce")

    # Medias de liga conocidas antes de cada kickoff.
    df["league_yellow_avg_before"] = yellow.shift(1).expanding(
        min_periods=1
    ).mean()
    df["league_fouls_avg_before"] = fouls.shift(1).expanding(
        min_periods=1
    ).mean()
    df["league_red_avg_before"] = red.shift(1).expanding(
        min_periods=1
    ).mean()

    # Prior inicial para las primeras filas sin historia.
    # Usamos SOLO un valor fijo neutral aproximado; en el entrenamiento las
    # primeras observaciones tienen poca influencia y no incorporan futuro.
    INITIAL_YELLOW_PRIOR = 4.5
    INITIAL_FOULS_PRIOR = 25.0
    INITIAL_RED_PRIOR = 0.20

    df["league_yellow_avg_before"] = df[
        "league_yellow_avg_before"
    ].fillna(INITIAL_YELLOW_PRIOR)
    df["league_fouls_avg_before"] = df[
        "league_fouls_avg_before"
    ].fillna(INITIAL_FOULS_PRIOR)
    df["league_red_avg_before"] = df[
        "league_red_avg_before"
    ].fillna(INITIAL_RED_PRIOR)

    # Variables del árbitro, siempre shift(1).
    featured = (
        df.groupby("referee", group_keys=False, dropna=False)
        .apply(referee_group_features, include_groups=False)
        .reset_index(drop=False)
    )

    # groupby/apply puede devolver referee como índice dependiendo de pandas.
    if "referee" not in featured.columns:
        featured = featured.rename(columns={"level_0": "referee"})

    # Recuperar columnas identificadoras si pandas las quitó con include_groups=False.
    # Más robusto: volver a unir por match_id desde df.
    id_cols = [
        "match_id",
        "season",
        "utc_time",
        "home_team",
        "away_team",
        "referee",
        "total_yellow",
        "total_red",
        "total_fouls",
        "league_yellow_avg_before",
        "league_fouls_avg_before",
        "league_red_avg_before",
    ]

    feature_cols = [
        c for c in featured.columns
        if c.startswith("ref_")
    ]

    featured = df[id_cols].merge(
        featured[["match_id"] + feature_cols],
        on="match_id",
        how="left",
        validate="one_to_one",
    )

    # Fallback neutral cuando el árbitro aún no tiene historial.
    for col in [
        "ref_yellow_avg_before",
        "ref_yellow_last5_before",
        "ref_yellow_last10_before",
        "ref_yellow_last20_before",
    ]:
        featured[col] = featured[col].fillna(
            featured["league_yellow_avg_before"]
        )

    for col in [
        "ref_fouls_avg_before",
        "ref_fouls_last5_before",
        "ref_fouls_last10_before",
        "ref_fouls_last20_before",
    ]:
        featured[col] = featured[col].fillna(
            featured["league_fouls_avg_before"]
        )

    for col in [
        "ref_red_avg_before",
        "ref_red_last5_before",
        "ref_red_last10_before",
        "ref_red_last20_before",
    ]:
        featured[col] = featured[col].fillna(
            featured["league_red_avg_before"]
        )

    featured["ref_matches_before"] = (
        pd.to_numeric(
            featured["ref_matches_before"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    n = featured["ref_matches_before"].astype(float)

    # Empirical-Bayes shrinkage: árbitros con 2 partidos no pueden pesar igual
    # que árbitros con 80.
    featured["ref_yellow_shrunk_before"] = (
        n * featured["ref_yellow_avg_before"]
        + PRIOR_STRENGTH * featured["league_yellow_avg_before"]
    ) / (n + PRIOR_STRENGTH)

    featured["ref_yellow_delta_before"] = (
        featured["ref_yellow_shrunk_before"]
        - featured["league_yellow_avg_before"]
    )

    featured["ref_fouls_shrunk_before"] = (
        n * featured["ref_fouls_avg_before"]
        + PRIOR_STRENGTH * featured["league_fouls_avg_before"]
    ) / (n + PRIOR_STRENGTH)

    featured["ref_fouls_delta_before"] = (
        featured["ref_fouls_shrunk_before"]
        - featured["league_fouls_avg_before"]
    )

    featured["ref_experience_log"] = np.log1p(
        featured["ref_matches_before"]
    )

    featured["ref_profile_before"] = featured[
        "ref_yellow_delta_before"
    ].apply(profile_from_delta)

    # Orden final por fecha.
    featured = featured.sort_values(
        ["utc_time", "match_id"]
    ).reset_index(drop=True)

    current = build_current_summary(df)

    # Convertimos timestamps a texto ISO antes de SQLite.
    featured["utc_time"] = featured["utc_time"].apply(
        lambda x: x.isoformat() if pd.notna(x) else None
    )

    with sqlite3.connect(DB_PATH) as conn:
        featured.to_sql(
            OUTPUT_TABLE,
            conn,
            if_exists="replace",
            index=False,
        )

        current.to_sql(
            CURRENT_TABLE,
            conn,
            if_exists="replace",
            index=False,
        )

        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{OUTPUT_TABLE}_match "
            f"ON {OUTPUT_TABLE}(match_id)"
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{OUTPUT_TABLE}_ref_time "
            f"ON {OUTPUT_TABLE}(referee, utc_time)"
        )
        conn.commit()

    print()
    print("=" * 92)
    print("FEATURES DE ARBITROS CREADAS")
    print("=" * 92)
    print(f"Tabla: {OUTPUT_TABLE}")
    print(f"Filas: {len(featured)}")
    print(f"Árbitros: {featured['referee'].nunique(dropna=True)}")
    print(
        "Partidos con >= 10 partidos previos del árbitro: "
        f"{int((featured['ref_matches_before'] >= 10).sum())}"
    )
    print(
        "Partidos con >= 20 partidos previos del árbitro: "
        f"{int((featured['ref_matches_before'] >= 20).sum())}"
    )

    print()
    print("TOP ÁRBITROS ACTUALES (media suavizada por tamaño de muestra):")
    print("-" * 92)

    top = current.sort_values(
        "shrunk_yellow",
        ascending=False,
    ).head(12)

    for r in top.itertuples(index=False):
        print(
            f"{r.referee:<36} | "
            f"N={int(r.matches):3d} | "
            f"raw={r.avg_yellow:4.2f} | "
            f"ajustada={r.shrunk_yellow:4.2f} | "
            f"delta={r.yellow_delta_vs_league:+4.2f} | "
            f"{r.profile}"
        )

    print()
    print("BOTTOM ÁRBITROS ACTUALES:")
    print("-" * 92)

    bottom = current.sort_values(
        "shrunk_yellow",
        ascending=True,
    ).head(8)

    for r in bottom.itertuples(index=False):
        print(
            f"{r.referee:<36} | "
            f"N={int(r.matches):3d} | "
            f"raw={r.avg_yellow:4.2f} | "
            f"ajustada={r.shrunk_yellow:4.2f} | "
            f"delta={r.yellow_delta_vs_league:+4.2f} | "
            f"{r.profile}"
        )

    print()
    print(
        "Siguiente paso: entrenar Cards V3 y comparar V2 vs V3 "
        "en walk-forward 2024/25, 2025/26 y 2026/27."
    )
    print("=" * 92)


if __name__ == "__main__":
    main()
