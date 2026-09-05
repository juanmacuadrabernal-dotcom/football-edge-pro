from __future__ import annotations

import math
import re
import sqlite3
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from prediction_engine_laliga_v3 import (
    PredictionEngineLaLiga as BasePredictionEngineLaLiga,
    DB_PATH,
    MODELS_DIR,
    model_columns,
    nb_over,
    estimate_nb_alpha,
)


CARDS_V3_MODEL = (
    MODELS_DIR / "laliga_cards_v3_referee_side_poisson.joblib"
)

REF_PRIOR_STRENGTH = 15.0


def _norm(value):
    if value is None or pd.isna(value):
        return ""

    s = str(value).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(
        ch for ch in s
        if not unicodedata.combining(ch)
    )
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _safe_mean(values):
    vals = [
        float(v)
        for v in values
        if v is not None and not pd.isna(v)
    ]
    return float(np.mean(vals)) if vals else np.nan


def _profile(delta):
    if delta is None or pd.isna(delta):
        return "SIN DATOS"
    if delta >= 0.60:
        return "MUY TARJETERO"
    if delta >= 0.25:
        return "TARJETERO"
    if delta <= -0.60:
        return "MUY PERMISIVO"
    if delta <= -0.25:
        return "PERMISIVO"
    return "NEUTRO"


class PredictionEngineLaLiga(BasePredictionEngineLaLiga):
    """
    V4 del motor LaLiga:
    - conserva goles, corners, SOT y fixtures de V8;
    - sustituye SOLO tarjetas por Cards V3 Referee;
    - resuelve aliases del árbitro;
    - calcula el impacto del árbitro frente a un escenario neutral.
    """

    def __init__(self):
        super().__init__()

        if not CARDS_V3_MODEL.exists():
            raise FileNotFoundError(
                "Falta el modelo Cards V3: "
                f"{CARDS_V3_MODEL.name}. "
                "Ejecuta primero: python laliga_cards_v3_referee.py"
            )

        self.cards_model = joblib.load(
            CARDS_V3_MODEL
        )
        self.cards_cols = model_columns(
            self.cards_model
        )

        self._ref_alias_map = (
            self._build_ref_alias_map()
        )

    def _build_ref_alias_map(self):
        if self.ref_history.empty:
            return {}

        candidates = {}

        for row in self.ref_history.itertuples(
            index=False
        ):
            canonical = getattr(
                row,
                "referee",
                None,
            )

            if canonical is None or pd.isna(canonical):
                continue

            canonical = str(canonical).strip()

            raw = getattr(
                row,
                "referee_raw",
                None,
            )

            for value in (canonical, raw):
                if value is None or pd.isna(value):
                    continue

                key = _norm(value)

                if not key:
                    continue

                candidates.setdefault(
                    key,
                    set(),
                ).add(canonical)

        alias_map = {}

        for key, names in candidates.items():
            if len(names) == 1:
                alias_map[key] = next(
                    iter(names)
                )

        return alias_map

    def _canonical_referee(self, referee):
        if referee is None or pd.isna(referee):
            return None

        raw = str(referee).strip()
        key = _norm(raw)

        if key in self._ref_alias_map:
            return self._ref_alias_map[key]

        # Exact canonical match, ignoring accents/case.
        if not self.ref_history.empty:
            canonical_values = (
                self.ref_history["referee"]
                .dropna()
                .astype(str)
                .unique()
            )

            matches = [
                x
                for x in canonical_values
                if _norm(x) == key
            ]

            if len(matches) == 1:
                return matches[0]

        return raw

    def _current_ref_features(
        self,
        referee,
        cutoff,
    ):
        prior = self.ref_history[
            self.ref_history["utc_time"] < cutoff
        ].copy()

        if prior.empty:
            league_y = 4.5
            league_f = 25.0
            league_r = 0.20
        else:
            league_y = float(
                pd.to_numeric(
                    prior["total_yellow"],
                    errors="coerce",
                ).mean()
            )
            league_f = float(
                pd.to_numeric(
                    prior["total_fouls"],
                    errors="coerce",
                ).mean()
            )
            league_r = float(
                pd.to_numeric(
                    prior["total_red"],
                    errors="coerce",
                ).mean()
            )

        canonical = self._canonical_referee(
            referee
        )

        if (
            canonical is None
            or prior.empty
        ):
            h = pd.DataFrame()
        else:
            h = prior[
                prior["referee"].astype(str)
                == str(canonical)
            ].sort_values(
                ["utc_time", "match_id"]
            )

        n = int(len(h))

        def stats(col, league):
            if h.empty or col not in h.columns:
                return {
                    "avg": league,
                    "last5": league,
                    "last10": league,
                    "last20": league,
                    "shrunk": league,
                }

            s = pd.to_numeric(
                h[col],
                errors="coerce",
            ).dropna()

            if s.empty:
                return {
                    "avg": league,
                    "last5": league,
                    "last10": league,
                    "last20": league,
                    "shrunk": league,
                }

            avg = float(s.mean())

            shrunk = float(
                (
                    n * avg
                    + REF_PRIOR_STRENGTH * league
                )
                / (
                    n
                    + REF_PRIOR_STRENGTH
                )
            )

            return {
                "avg": avg,
                "last5": float(s.tail(5).mean()),
                "last10": float(s.tail(10).mean()),
                "last20": float(s.tail(20).mean()),
                "shrunk": shrunk,
            }

        y = stats(
            "total_yellow",
            league_y,
        )
        f = stats(
            "total_fouls",
            league_f,
        )
        r = stats(
            "total_red",
            league_r,
        )

        delta_y = (
            y["shrunk"] - league_y
        )
        delta_f = (
            f["shrunk"] - league_f
        )

        return {
            "canonical_referee": canonical,
            "ref_matches_before": n,

            "ref_yellow_avg_before": y["avg"],
            "ref_yellow_last5_before": y["last5"],
            "ref_yellow_last10_before": y["last10"],
            "ref_yellow_last20_before": y["last20"],
            "ref_yellow_shrunk_before": y["shrunk"],
            "ref_yellow_delta_before": delta_y,

            "ref_fouls_avg_before": f["avg"],
            "ref_fouls_last5_before": f["last5"],
            "ref_fouls_last10_before": f["last10"],
            "ref_fouls_last20_before": f["last20"],
            "ref_fouls_shrunk_before": f["shrunk"],
            "ref_fouls_delta_before": delta_f,

            "ref_red_avg_before": r["avg"],
            "ref_red_last5_before": r["last5"],
            "ref_red_last10_before": r["last10"],
            "ref_red_last20_before": r["last20"],

            "ref_experience_log": float(
                np.log1p(n)
            ),

            "league_yellow_avg_before": league_y,
            "league_fouls_avg_before": league_f,
            "league_red_avg_before": league_r,

            # Datos para mostrar en interfaz.
            "_raw_avg_yellow": y["avg"],
            "_last10_yellow": y["last10"],
            "_last20_yellow": y["last20"],
            "_profile": _profile(delta_y),
        }

    def _neutral_ref_features(self, actual):
        neutral = dict(actual)

        league_y = actual[
            "league_yellow_avg_before"
        ]
        league_f = actual[
            "league_fouls_avg_before"
        ]
        league_r = actual[
            "league_red_avg_before"
        ]

        for key in (
            "ref_yellow_avg_before",
            "ref_yellow_last5_before",
            "ref_yellow_last10_before",
            "ref_yellow_last20_before",
            "ref_yellow_shrunk_before",
        ):
            neutral[key] = league_y

        neutral[
            "ref_yellow_delta_before"
        ] = 0.0

        for key in (
            "ref_fouls_avg_before",
            "ref_fouls_last5_before",
            "ref_fouls_last10_before",
            "ref_fouls_last20_before",
            "ref_fouls_shrunk_before",
        ):
            neutral[key] = league_f

        neutral[
            "ref_fouls_delta_before"
        ] = 0.0

        for key in (
            "ref_red_avg_before",
            "ref_red_last5_before",
            "ref_red_last10_before",
            "ref_red_last20_before",
        ):
            neutral[key] = league_r

        neutral["_profile"] = "NEUTRO"

        return neutral

    def _build_cards_v3_row(
        self,
        team,
        opponent,
        side,
        referee_features,
        cutoff,
        season,
    ):
        league = self._league_cards_features(
            cutoff,
            season,
            side,
        )

        row = {}

        for col in self.cards_cols:
            if col == "side":
                row[col] = side
                continue

            if col == "team":
                row[col] = team
                continue

            if col == "opponent":
                row[col] = opponent
                continue

            if col == "referee":
                row[col] = (
                    referee_features.get(
                        "canonical_referee"
                    )
                    or np.nan
                )
                continue

            if col in referee_features:
                row[col] = referee_features[col]
                continue

            if col in league:
                row[col] = league[col]
                continue

            m = re.fullmatch(
                r"team_yellow_for_(last5|last10|season)",
                col,
            )
            if m:
                row[col] = self._team_mean(
                    team,
                    "yellow_for",
                    m.group(1),
                    cutoff,
                    season,
                )
                continue

            m = re.fullmatch(
                r"opp_yellow_against_(last5|last10|season)",
                col,
            )
            if m:
                row[col] = self._team_mean(
                    opponent,
                    "yellow_against",
                    m.group(1),
                    cutoff,
                    season,
                )
                continue

            m = re.fullmatch(
                r"team_fouls_for_(last5|last10|season)",
                col,
            )
            if m:
                row[col] = self._team_mean(
                    team,
                    "fouls_for",
                    m.group(1),
                    cutoff,
                    season,
                )
                continue

            m = re.fullmatch(
                r"opp_fouls_against_(last5|last10|season)",
                col,
            )
            if m:
                row[col] = self._team_mean(
                    opponent,
                    "fouls_against",
                    m.group(1),
                    cutoff,
                    season,
                )
                continue

            m = re.fullmatch(
                r"expected_cards_v2_(last5|last10|season)",
                col,
            )
            if m:
                team_y = self._team_mean(
                    team,
                    "yellow_for",
                    m.group(1),
                    cutoff,
                    season,
                )

                opp_y = self._team_mean(
                    opponent,
                    "yellow_against",
                    m.group(1),
                    cutoff,
                    season,
                )

                row[col] = _safe_mean(
                    [team_y, opp_y]
                )
                continue

            m = re.fullmatch(
                r"expected_cards_v3_(last5|last10|season)",
                col,
            )
            if m:
                team_y = self._team_mean(
                    team,
                    "yellow_for",
                    m.group(1),
                    cutoff,
                    season,
                )

                opp_y = self._team_mean(
                    opponent,
                    "yellow_against",
                    m.group(1),
                    cutoff,
                    season,
                )

                ref_total = referee_features.get(
                    "ref_yellow_shrunk_before",
                    np.nan,
                )

                ref_side = (
                    float(ref_total) / 2.0
                    if ref_total is not None
                    and not pd.isna(ref_total)
                    else np.nan
                )

                row[col] = _safe_mean(
                    [team_y, opp_y, ref_side]
                )
                continue

            # Compatibilidad por si se regenera el V3 con nombres antiguos.
            m = re.fullmatch(
                r"expected_cards_(last5|last10|season)",
                col,
            )
            if m:
                team_y = self._team_mean(
                    team,
                    "yellow_for",
                    m.group(1),
                    cutoff,
                    season,
                )

                opp_y = self._team_mean(
                    opponent,
                    "yellow_against",
                    m.group(1),
                    cutoff,
                    season,
                )

                ref_total = referee_features.get(
                    "ref_yellow_shrunk_before",
                    np.nan,
                )

                ref_side = (
                    float(ref_total) / 2.0
                    if ref_total is not None
                    and not pd.isna(ref_total)
                    else np.nan
                )

                row[col] = _safe_mean(
                    [team_y, opp_y, ref_side]
                )
                continue

            row[col] = np.nan

        return pd.DataFrame(
            [row],
            columns=self.cards_cols,
        )

    def _latest_v3_weight(
        self,
        market,
        fallback=0.75,
    ):
        try:
            with sqlite3.connect(DB_PATH) as conn:
                q = pd.read_sql_query(
                    """
                    SELECT season, v3_shrink_weight
                    FROM cards_v3_referee_backtest_predictions
                    WHERE market = ?
                      AND v3_shrink_weight IS NOT NULL
                    ORDER BY season DESC
                    LIMIT 1
                    """,
                    conn,
                    params=(market,),
                )

            if not q.empty:
                return float(
                    q.iloc[0][
                        "v3_shrink_weight"
                    ]
                )
        except Exception:
            pass

        return float(fallback)

    def predict_cards(self, fixture):
        cutoff = fixture["utc_time"]
        season = int(fixture["season"])

        raw_referee = fixture.get(
            "referee",
            np.nan,
        )

        ref_actual = self._current_ref_features(
            raw_referee,
            cutoff,
        )

        ref_neutral = self._neutral_ref_features(
            ref_actual
        )

        home_actual = self._build_cards_v3_row(
            fixture["home_team"],
            fixture["away_team"],
            "home",
            ref_actual,
            cutoff,
            season,
        )

        away_actual = self._build_cards_v3_row(
            fixture["away_team"],
            fixture["home_team"],
            "away",
            ref_actual,
            cutoff,
            season,
        )

        home_neutral = self._build_cards_v3_row(
            fixture["home_team"],
            fixture["away_team"],
            "home",
            ref_neutral,
            cutoff,
            season,
        )

        away_neutral = self._build_cards_v3_row(
            fixture["away_team"],
            fixture["home_team"],
            "away",
            ref_neutral,
            cutoff,
            season,
        )

        mu_home = float(
            np.clip(
                self.cards_model.predict(
                    home_actual
                )[0],
                0.05,
                5.0,
            )
        )

        mu_away = float(
            np.clip(
                self.cards_model.predict(
                    away_actual
                )[0],
                0.05,
                5.0,
            )
        )

        mu_neutral_home = float(
            np.clip(
                self.cards_model.predict(
                    home_neutral
                )[0],
                0.05,
                5.0,
            )
        )

        mu_neutral_away = float(
            np.clip(
                self.cards_model.predict(
                    away_neutral
                )[0],
                0.05,
                5.0,
            )
        )

        mu_total = mu_home + mu_away
        mu_neutral = (
            mu_neutral_home
            + mu_neutral_away
        )

        match_hist = self._match_level_history(
            cutoff
        )

        if match_hist.empty:
            baseline = mu_total
            alpha = 0.10
        else:
            totals = (
                pd.to_numeric(
                    match_hist["yellow_for"],
                    errors="coerce",
                )
                + pd.to_numeric(
                    match_hist["yellow_against"],
                    errors="coerce",
                )
            ).dropna()

            baseline = float(
                totals.tail(100).mean()
            )

            alpha = estimate_nb_alpha(
                totals
            )

        assigned = (
            raw_referee is not None
            and not pd.isna(raw_referee)
            and str(raw_referee).strip() != ""
        )

        referee_name = (
            ref_actual.get(
                "canonical_referee"
            )
            if assigned
            else None
        )

        output = {
            "model_version": "Cards V3 Referee",
            "mu_home": mu_home,
            "mu_away": mu_away,
            "mu_total": mu_total,
            "mu_neutral": mu_neutral,
            "referee_effect_mu": (
                mu_total - mu_neutral
                if assigned
                else 0.0
            ),
            "nb_alpha": alpha,
            "referee_info": {
                "assigned": bool(assigned),
                "name": referee_name,
                "matches": int(
                    ref_actual.get(
                        "ref_matches_before",
                        0,
                    )
                ),
                "avg_yellow": float(
                    ref_actual.get(
                        "_raw_avg_yellow",
                        np.nan,
                    )
                ),
                "shrunk_yellow": float(
                    ref_actual.get(
                        "ref_yellow_shrunk_before",
                        np.nan,
                    )
                ),
                "last10_yellow": float(
                    ref_actual.get(
                        "_last10_yellow",
                        np.nan,
                    )
                ),
                "last20_yellow": float(
                    ref_actual.get(
                        "_last20_yellow",
                        np.nan,
                    )
                ),
                "league_avg_yellow": float(
                    ref_actual.get(
                        "league_yellow_avg_before",
                        np.nan,
                    )
                ),
                "delta_vs_league": float(
                    ref_actual.get(
                        "ref_yellow_delta_before",
                        0.0,
                    )
                ),
                "profile": (
                    ref_actual.get(
                        "_profile",
                        "SIN DATOS",
                    )
                    if assigned
                    else "PENDIENTE"
                ),
            },
            "referee_effect_probability": {},
            "neutral_probability": {},
        }

        for line in (
            2.5,
            3.5,
            4.5,
            5.5,
        ):
            market = (
                f"over_"
                f"{str(line).replace('.', '_')}"
            )

            raw_actual = nb_over(
                line,
                mu_total,
                alpha,
            )

            raw_neutral = nb_over(
                line,
                mu_neutral,
                alpha,
            )

            base = nb_over(
                line,
                baseline,
                alpha,
            )

            weight = self._latest_v3_weight(
                market,
                0.75,
            )

            p_actual = float(
                np.clip(
                    weight * raw_actual
                    + (1 - weight) * base,
                    0,
                    1,
                )
            )

            p_neutral = float(
                np.clip(
                    weight * raw_neutral
                    + (1 - weight) * base,
                    0,
                    1,
                )
            )

            output[market] = p_actual
            output[
                "neutral_probability"
            ][market] = p_neutral

            output[
                "referee_effect_probability"
            ][market] = (
                p_actual - p_neutral
                if assigned
                else 0.0
            )

        return output
