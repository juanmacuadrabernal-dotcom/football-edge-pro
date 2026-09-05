from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "eliteserien.db"

ROLLING_WINDOWS = (5, 10)

METRICS = [
    "goals_for",
    "goals_against",
    "xg_for",
    "xg_against",
    "shots_for",
    "shots_against",
    "shots_on_target_for",
    "shots_on_target_against",
    "corners_for",
    "corners_against",
    "fouls_for",
    "fouls_against",
    "offsides_for",
    "offsides_against",
    "yellow_for",
    "yellow_against",
    "red_for",
    "red_against",
    "points",
    "win",
    "draw",
    "loss",
    "btts",
    "over_15",
    "over_25",
    "over_35",
]


def load_matches(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Solo usamos partidos:
    - terminados
    - no cancelados
    - válidos para modelo
    - con marcador
    """

    query = """
    SELECT
        m.match_id,
        m.season,
        m.round,
        m.utc_time,
        m.home_team_id,
        m.home_team,
        m.away_team_id,
        m.away_team,
        m.home_goals,
        m.away_goals,
        m.finished,
        m.cancelled,
        m.valid_for_model,

        s.home_xg,
        s.away_xg,
        s.home_possession,
        s.away_possession,
        s.home_shots,
        s.away_shots,
        s.home_shots_on_target,
        s.away_shots_on_target,
        s.home_corners,
        s.away_corners,
        s.home_fouls,
        s.away_fouls,
        s.home_offsides,
        s.away_offsides,
        s.home_yellow_cards,
        s.away_yellow_cards,
        s.home_red_cards,
        s.away_red_cards,
        s.referee,
        s.stadium,
        s.attendance

    FROM fotmob_matches m
    LEFT JOIN fotmob_match_stats s
        ON s.match_id = m.match_id

    WHERE m.finished = 1
      AND m.cancelled = 0
      AND COALESCE(m.valid_for_model, 1) = 1
      AND m.home_goals IS NOT NULL
      AND m.away_goals IS NOT NULL

    ORDER BY m.utc_time, m.match_id
    """

    df = pd.read_sql_query(query, conn)

    if df.empty:
        raise RuntimeError("No hay partidos válidos en la base de datos.")

    df["utc_time"] = pd.to_datetime(df["utc_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["utc_time"]).copy()

    return df


def build_team_history(matches: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for r in matches.itertuples(index=False):
        total_goals = float(r.home_goals + r.away_goals)
        btts = int(r.home_goals > 0 and r.away_goals > 0)

        # -------------------------
        # LOCAL
        # -------------------------
        if r.home_goals > r.away_goals:
            home_result = "W"
            home_points = 3
            home_win, home_draw, home_loss = 1, 0, 0
        elif r.home_goals == r.away_goals:
            home_result = "D"
            home_points = 1
            home_win, home_draw, home_loss = 0, 1, 0
        else:
            home_result = "L"
            home_points = 0
            home_win, home_draw, home_loss = 0, 0, 1

        rows.append({
            "match_id": r.match_id,
            "season": r.season,
            "round": r.round,
            "utc_time": r.utc_time,
            "team_id": r.home_team_id,
            "team": r.home_team,
            "opponent_id": r.away_team_id,
            "opponent": r.away_team,
            "venue": "home",
            "is_home": 1,

            "goals_for": r.home_goals,
            "goals_against": r.away_goals,

            "xg_for": r.home_xg,
            "xg_against": r.away_xg,

            "possession": r.home_possession,

            "shots_for": r.home_shots,
            "shots_against": r.away_shots,

            "shots_on_target_for": r.home_shots_on_target,
            "shots_on_target_against": r.away_shots_on_target,

            "corners_for": r.home_corners,
            "corners_against": r.away_corners,

            "fouls_for": r.home_fouls,
            "fouls_against": r.away_fouls,

            "offsides_for": r.home_offsides,
            "offsides_against": r.away_offsides,

            "yellow_for": r.home_yellow_cards,
            "yellow_against": r.away_yellow_cards,

            "red_for": r.home_red_cards,
            "red_against": r.away_red_cards,

            "cards_events_for": (
                (r.home_yellow_cards if pd.notna(r.home_yellow_cards) else 0)
                + (r.home_red_cards if pd.notna(r.home_red_cards) else 0)
            ),
            "cards_events_against": (
                (r.away_yellow_cards if pd.notna(r.away_yellow_cards) else 0)
                + (r.away_red_cards if pd.notna(r.away_red_cards) else 0)
            ),

            "result": home_result,
            "points": home_points,
            "win": home_win,
            "draw": home_draw,
            "loss": home_loss,

            "btts": btts,
            "over_15": int(total_goals > 1.5),
            "over_25": int(total_goals > 2.5),
            "over_35": int(total_goals > 3.5),

            "referee": r.referee,
            "stadium": r.stadium,
            "attendance": r.attendance,
        })

        # -------------------------
        # VISITANTE
        # -------------------------
        if r.away_goals > r.home_goals:
            away_result = "W"
            away_points = 3
            away_win, away_draw, away_loss = 1, 0, 0
        elif r.away_goals == r.home_goals:
            away_result = "D"
            away_points = 1
            away_win, away_draw, away_loss = 0, 1, 0
        else:
            away_result = "L"
            away_points = 0
            away_win, away_draw, away_loss = 0, 0, 1

        rows.append({
            "match_id": r.match_id,
            "season": r.season,
            "round": r.round,
            "utc_time": r.utc_time,
            "team_id": r.away_team_id,
            "team": r.away_team,
            "opponent_id": r.home_team_id,
            "opponent": r.home_team,
            "venue": "away",
            "is_home": 0,

            "goals_for": r.away_goals,
            "goals_against": r.home_goals,

            "xg_for": r.away_xg,
            "xg_against": r.home_xg,

            "possession": r.away_possession,

            "shots_for": r.away_shots,
            "shots_against": r.home_shots,

            "shots_on_target_for": r.away_shots_on_target,
            "shots_on_target_against": r.home_shots_on_target,

            "corners_for": r.away_corners,
            "corners_against": r.home_corners,

            "fouls_for": r.away_fouls,
            "fouls_against": r.home_fouls,

            "offsides_for": r.away_offsides,
            "offsides_against": r.home_offsides,

            "yellow_for": r.away_yellow_cards,
            "yellow_against": r.home_yellow_cards,

            "red_for": r.away_red_cards,
            "red_against": r.home_red_cards,

            "cards_events_for": (
                (r.away_yellow_cards if pd.notna(r.away_yellow_cards) else 0)
                + (r.away_red_cards if pd.notna(r.away_red_cards) else 0)
            ),
            "cards_events_against": (
                (r.home_yellow_cards if pd.notna(r.home_yellow_cards) else 0)
                + (r.home_red_cards if pd.notna(r.home_red_cards) else 0)
            ),

            "result": away_result,
            "points": away_points,
            "win": away_win,
            "draw": away_draw,
            "loss": away_loss,

            "btts": btts,
            "over_15": int(total_goals > 1.5),
            "over_25": int(total_goals > 2.5),
            "over_35": int(total_goals > 3.5),

            "referee": r.referee,
            "stadium": r.stadium,
            "attendance": r.attendance,
        })

    history = pd.DataFrame(rows)
    history = history.sort_values(
        ["team", "utc_time", "match_id"]
    ).reset_index(drop=True)

    return history


def add_pre_match_features(history: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features PRE-PARTIDO.

    MUY IMPORTANTE:
    Todo usa shift(1), así que el partido actual NO participa
    en sus propias medias. Esto evita data leakage.
    """

    df = history.copy()

    feature_columns = {}

    for team, idx in df.groupby("team", sort=False).groups.items():
        team_df = df.loc[idx].sort_values(["utc_time", "match_id"])

        # Número de partido histórico antes de jugar.
        feature_columns.setdefault("history_matches_before", pd.Series(index=df.index, dtype=float))
        feature_columns["history_matches_before"].loc[team_df.index] = np.arange(len(team_df))

        # Número de partido de temporada antes de jugar.
        feature_columns.setdefault("season_matches_before", pd.Series(index=df.index, dtype=float))
        season_counts = team_df.groupby("season").cumcount()
        feature_columns["season_matches_before"].loc[team_df.index] = season_counts.values

        # -------------------------------------------------
        # Últimos 5 / 10 - cualquier campo
        # -------------------------------------------------
        for metric in METRICS:
            values = pd.to_numeric(team_df[metric], errors="coerce")

            for window in ROLLING_WINDOWS:
                col = f"pre_last{window}_{metric}"
                feature_columns.setdefault(col, pd.Series(index=df.index, dtype=float))

                rolled = (
                    values
                    .shift(1)
                    .rolling(window=window, min_periods=1)
                    .mean()
                )

                feature_columns[col].loc[team_df.index] = rolled.values

        # -------------------------------------------------
        # Temporada actual hasta antes del partido
        # -------------------------------------------------
        for metric in METRICS:
            col = f"pre_season_{metric}"
            feature_columns.setdefault(col, pd.Series(index=df.index, dtype=float))

            result = pd.Series(index=team_df.index, dtype=float)

            for season, season_idx in team_df.groupby("season", sort=False).groups.items():
                season_slice = team_df.loc[season_idx].sort_values(["utc_time", "match_id"])
                values = pd.to_numeric(season_slice[metric], errors="coerce")

                expanding = (
                    values
                    .shift(1)
                    .expanding(min_periods=1)
                    .mean()
                )

                result.loc[season_slice.index] = expanding.values

            feature_columns[col].loc[team_df.index] = result.loc[team_df.index].values

        # -------------------------------------------------
        # Local/visitante: últimos 5 en el mismo venue
        # -------------------------------------------------
        for venue in ("home", "away"):
            venue_df = team_df[team_df["venue"] == venue]

            for metric in [
                "goals_for",
                "goals_against",
                "xg_for",
                "xg_against",
                "shots_for",
                "shots_against",
                "shots_on_target_for",
                "shots_on_target_against",
                "corners_for",
                "corners_against",
                "yellow_for",
                "yellow_against",
                "points",
                "win",
                "btts",
                "over_25",
            ]:
                col = f"pre_{venue}_last5_{metric}"
                feature_columns.setdefault(col, pd.Series(index=df.index, dtype=float))

                values = pd.to_numeric(venue_df[metric], errors="coerce")

                rolled = (
                    values
                    .shift(1)
                    .rolling(window=5, min_periods=1)
                    .mean()
                )

                feature_columns[col].loc[venue_df.index] = rolled.values

    # Añadimos todas las columnas de golpe.
    for col, series in feature_columns.items():
        df[col] = series

    return df


def build_current_team_summary(history: pd.DataFrame) -> pd.DataFrame:
    rows = []

    current_season = int(history["season"].max())

    for team, group in history.groupby("team"):
        g = group.sort_values(["utc_time", "match_id"]).copy()
        current = g[g["season"] == current_season].copy()

        if current.empty:
            current = g.tail(10).copy()

        last5 = g.tail(5)
        last10 = g.tail(10)

        def avg(frame, col):
            if frame.empty or col not in frame:
                return None
            value = pd.to_numeric(frame[col], errors="coerce").mean()
            return None if pd.isna(value) else float(value)

        row = {
            "team": team,
            "team_id": g["team_id"].dropna().iloc[-1] if g["team_id"].notna().any() else None,
            "current_season": current_season,
            "last_match": g["utc_time"].max().isoformat(),
            "matches_all": len(g),
            "matches_current_season": len(current),

            "last5_points_pg": avg(last5, "points"),
            "last10_points_pg": avg(last10, "points"),

            "last5_goals_for": avg(last5, "goals_for"),
            "last5_goals_against": avg(last5, "goals_against"),
            "last10_goals_for": avg(last10, "goals_for"),
            "last10_goals_against": avg(last10, "goals_against"),

            "last5_xg_for": avg(last5, "xg_for"),
            "last5_xg_against": avg(last5, "xg_against"),
            "last10_xg_for": avg(last10, "xg_for"),
            "last10_xg_against": avg(last10, "xg_against"),

            "last5_corners_for": avg(last5, "corners_for"),
            "last5_corners_against": avg(last5, "corners_against"),
            "last10_corners_for": avg(last10, "corners_for"),
            "last10_corners_against": avg(last10, "corners_against"),

            "last5_yellow_for": avg(last5, "yellow_for"),
            "last5_yellow_against": avg(last5, "yellow_against"),
            "last10_yellow_for": avg(last10, "yellow_for"),
            "last10_yellow_against": avg(last10, "yellow_against"),

            "season_goals_for": avg(current, "goals_for"),
            "season_goals_against": avg(current, "goals_against"),
            "season_xg_for": avg(current, "xg_for"),
            "season_xg_against": avg(current, "xg_against"),
            "season_corners_for": avg(current, "corners_for"),
            "season_corners_against": avg(current, "corners_against"),
            "season_yellow_for": avg(current, "yellow_for"),
            "season_yellow_against": avg(current, "yellow_against"),
            "season_points_pg": avg(current, "points"),
        }

        rows.append(row)

    return pd.DataFrame(rows).sort_values("team").reset_index(drop=True)


def build_league_summary(history: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # Cada partido aparece dos veces en team history.
    # Para métricas "for", la media por equipo/partido es válida.
    for season, g in history.groupby("season"):
        rows.append({
            "season": int(season),
            "team_match_rows": len(g),
            "matches": int(g["match_id"].nunique()),
            "avg_goals_per_team": pd.to_numeric(g["goals_for"], errors="coerce").mean(),
            "avg_xg_per_team": pd.to_numeric(g["xg_for"], errors="coerce").mean(),
            "avg_shots_per_team": pd.to_numeric(g["shots_for"], errors="coerce").mean(),
            "avg_shots_on_target_per_team": pd.to_numeric(
                g["shots_on_target_for"], errors="coerce"
            ).mean(),
            "avg_corners_per_team": pd.to_numeric(g["corners_for"], errors="coerce").mean(),
            "avg_yellow_per_team": pd.to_numeric(g["yellow_for"], errors="coerce").mean(),
            "avg_red_per_team": pd.to_numeric(g["red_for"], errors="coerce").mean(),
            "btts_rate": (
                g.drop_duplicates("match_id")["btts"].mean()
            ),
            "over25_rate": (
                g.drop_duplicates("match_id")["over_25"].mean()
            ),
        })

    return pd.DataFrame(rows).sort_values("season").reset_index(drop=True)


def save_tables(
    conn: sqlite3.Connection,
    history: pd.DataFrame,
    features: pd.DataFrame,
    current_summary: pd.DataFrame,
    league_summary: pd.DataFrame,
):
    # SQLite no entiende datetime timezone directamente.
    for frame in (history, features):
        frame["utc_time"] = frame["utc_time"].astype(str)

    history.to_sql(
        "team_match_history",
        conn,
        if_exists="replace",
        index=False,
    )

    features.to_sql(
        "team_pre_match_features",
        conn,
        if_exists="replace",
        index=False,
    )

    current_summary.to_sql(
        "team_current_summary",
        conn,
        if_exists="replace",
        index=False,
    )

    league_summary.to_sql(
        "league_season_summary",
        conn,
        if_exists="replace",
        index=False,
    )

    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_tmh_team_time
        ON team_match_history(team, utc_time);

        CREATE INDEX IF NOT EXISTS idx_tmh_match
        ON team_match_history(match_id);

        CREATE INDEX IF NOT EXISTS idx_tpmf_team_time
        ON team_pre_match_features(team, utc_time);

        CREATE INDEX IF NOT EXISTS idx_tpmf_match
        ON team_pre_match_features(match_id);
        """
    )

    conn.commit()


def validate(history, features):
    expected_rows = history["match_id"].nunique() * 2

    if len(history) != expected_rows:
        raise RuntimeError(
            f"Esperábamos {expected_rows} filas de equipo-partido "
            f"y hay {len(history)}."
        )

    # Comprobación simple de leakage:
    # primer partido histórico de cada equipo debe no tener last5 previo.
    first_rows = (
        features.sort_values(["team", "utc_time", "match_id"])
        .groupby("team", as_index=False)
        .head(1)
    )

    leakage_count = first_rows["pre_last5_goals_for"].notna().sum()

    if leakage_count:
        raise RuntimeError(
            "Detectado posible data leakage en las medias pre-partido."
        )


def main():
    print()
    print("=" * 72)
    print("ELITESERIEN EDGE PRO - CONSTRUIR ESTADISTICAS DE EQUIPOS")
    print("=" * 72)

    if not DB_PATH.exists():
        print(f"ERROR: no existe la base de datos: {DB_PATH}")
        raise SystemExit(1)

    with sqlite3.connect(DB_PATH) as conn:
        print("1/5 Cargando partidos limpios...")
        matches = load_matches(conn)

        print(f"    Partidos válidos: {len(matches)}")

        print("2/5 Creando historial por equipo...")
        history = build_team_history(matches)

        print(f"    Filas equipo-partido: {len(history)}")

        print("3/5 Calculando últimos 5/10, temporada y local/visitante...")
        features = add_pre_match_features(history)

        print("4/5 Creando resumen actual por equipo y liga...")
        current_summary = build_current_team_summary(history)
        league_summary = build_league_summary(history)

        print("5/5 Validando y guardando en SQLite...")
        validate(history, features)

        save_tables(
            conn,
            history,
            features,
            current_summary,
            league_summary,
        )

    print()
    print("=" * 72)
    print("ESTADISTICAS DE EQUIPOS CREADAS CORRECTAMENTE")
    print("=" * 72)
    print(f"Partidos usados: {matches['match_id'].nunique()}")
    print(f"Filas equipo-partido: {len(history)}")
    print(f"Equipos históricos: {history['team'].nunique()}")
    print(f"Temporadas: {sorted(history['season'].unique().tolist())}")
    print()
    print("Tablas creadas:")
    print("  team_match_history")
    print("  team_pre_match_features")
    print("  team_current_summary")
    print("  league_season_summary")
    print()
    print("Features PRE-PARTIDO:")
    print("  últimos 5")
    print("  últimos 10")
    print("  temporada hasta ese partido")
    print("  últimos 5 como local")
    print("  últimos 5 como visitante")
    print()
    print("Mercados preparados:")
    print("  goles")
    print("  BTTS")
    print("  over 1.5 / 2.5 / 3.5")
    print("  xG")
    print("  tiros")
    print("  tiros a puerta")
    print("  corners")
    print("  tarjetas")
    print("  faltas")
    print("  offsides")
    print("=" * 72)


if __name__ == "__main__":
    main()
