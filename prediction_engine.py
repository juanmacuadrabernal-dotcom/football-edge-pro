from __future__ import annotations

import math
import re
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "eliteserien.db"
MODELS_DIR = BASE_DIR / "models"

GOALS_MODEL = MODELS_DIR / "goals_v4_team_strength_poisson.joblib"
CORNERS_MODEL = MODELS_DIR / "corners_v1_poisson.joblib"
CARDS_MODEL = MODELS_DIR / "cards_v2_side_poisson.joblib"


def poisson_cdf(k: int, lam: float) -> float:
    lam = max(float(lam), 1e-8)
    term = math.exp(-lam)
    total = term
    for i in range(1, k + 1):
        term *= lam / i
        total += term
    return float(np.clip(total, 0, 1))


def poisson_over(line: float, lam: float) -> float:
    return 1.0 - poisson_cdf(int(math.floor(line)), lam)


def poisson_pmf(k: int, lam: float) -> float:
    lam = max(float(lam), 1e-8)
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def result_probs(lh: float, la: float, max_goals: int = 10):
    h = d = a = 0.0

    for hg in range(max_goals + 1):
        ph = poisson_pmf(hg, lh)
        for ag in range(max_goals + 1):
            p = ph * poisson_pmf(ag, la)
            if hg > ag:
                h += p
            elif hg == ag:
                d += p
            else:
                a += p

    total = h + d + a
    if total <= 0:
        return 1/3, 1/3, 1/3

    return h / total, d / total, a / total


def estimate_nb_alpha(values) -> float:
    y = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).dropna().to_numpy(dtype=float)

    if len(y) < 30:
        return 0.10

    mu = float(np.mean(y))
    var = float(np.var(y, ddof=1))

    if mu <= 0:
        return 0.10

    return float(
        np.clip(
            (var - mu) / (mu * mu),
            0.01,
            1.50,
        )
    )


def nb_cdf(k: int, mu: float, alpha: float) -> float:
    mu = max(float(mu), 1e-8)
    alpha = max(float(alpha), 1e-8)

    r = 1.0 / alpha
    p = r / (r + mu)

    total = 0.0

    for x in range(k + 1):
        log_pmf = (
            math.lgamma(x + r)
            - math.lgamma(r)
            - math.lgamma(x + 1)
            + r * math.log(p)
            + x * math.log(1 - p)
        )
        total += math.exp(log_pmf)

    return float(np.clip(total, 0, 1))


def nb_over(line: float, mu: float, alpha: float) -> float:
    return 1.0 - nb_cdf(
        int(math.floor(line)),
        mu,
        alpha,
    )


def pick_col(columns, candidates):
    lower = {c.lower(): c for c in columns}

    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    return None


def model_columns(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    prep = model.named_steps.get("prep")
    if prep is None:
        raise RuntimeError("No puedo detectar las features del modelo.")

    cols = []

    for _, _, selected in prep.transformers_:
        if isinstance(selected, (list, tuple, np.ndarray, pd.Index)):
            cols.extend(str(c) for c in selected)

    return list(dict.fromkeys(cols))


class PredictionEngine:
    def __init__(self):
        if not DB_PATH.exists():
            raise FileNotFoundError(f"No existe {DB_PATH}")

        for path in (GOALS_MODEL, CORNERS_MODEL, CARDS_MODEL):
            if not path.exists():
                raise FileNotFoundError(
                    f"Falta modelo: {path.name}"
                )

        self.goals_model = joblib.load(GOALS_MODEL)
        self.corners_model = joblib.load(CORNERS_MODEL)
        self.cards_model = joblib.load(CARDS_MODEL)

        self.goals_cols = model_columns(self.goals_model)
        self.corners_cols = model_columns(self.corners_model)
        self.cards_cols = model_columns(self.cards_model)

        with sqlite3.connect(DB_PATH) as conn:
            self.history = pd.read_sql_query(
                """
                SELECT *
                FROM team_match_history
                ORDER BY utc_time, match_id
                """,
                conn,
            )

            try:
                self.ref_history = pd.read_sql_query(
                    """
                    SELECT *
                    FROM referee_match_history
                    ORDER BY utc_time, match_id
                    """,
                    conn,
                )
            except Exception:
                self.ref_history = pd.DataFrame()

            self.fixtures_raw = pd.read_sql_query(
                """
                SELECT *
                FROM fotmob_matches
                """,
                conn,
            )

        self._prepare_history()
        self._prepare_ref_history()
        self.fixtures = self._prepare_fixtures(self.fixtures_raw)

    def _prepare_history(self):
        self.history["utc_time"] = pd.to_datetime(
            self.history["utc_time"],
            utc=True,
            errors="coerce",
        )

        self.history["season"] = pd.to_numeric(
            self.history["season"],
            errors="coerce",
        )

        self.history = self.history.dropna(
            subset=["utc_time", "team"]
        ).copy()

    def _prepare_ref_history(self):
        if self.ref_history.empty:
            return

        self.ref_history["utc_time"] = pd.to_datetime(
            self.ref_history["utc_time"],
            utc=True,
            errors="coerce",
        )

        self.ref_history = self.ref_history.dropna(
            subset=["utc_time"]
        ).copy()

    def _prepare_fixtures(self, raw):
        cols = list(raw.columns)

        mapping = {
            "match_id": pick_col(cols, ["match_id", "id"]),
            "season": pick_col(cols, ["season"]),
            "round": pick_col(
                cols,
                ["round", "round_name", "league_round"],
            ),
            "utc_time": pick_col(
                cols,
                ["utc_time", "date", "match_time", "kickoff"],
            ),
            "home_team": pick_col(
                cols,
                ["home_team", "home"],
            ),
            "away_team": pick_col(
                cols,
                ["away_team", "away"],
            ),
            "status": pick_col(
                cols,
                ["status", "status_text"],
            ),
            "referee": pick_col(
                cols,
                ["referee"],
            ),
            "valid_for_model": pick_col(
                cols,
                ["valid_for_model"],
            ),
        }

        required = [
            "match_id",
            "season",
            "utc_time",
            "home_team",
            "away_team",
        ]

        missing = [
            key for key in required
            if mapping[key] is None
        ]

        if missing:
            raise RuntimeError(
                "No puedo detectar columnas de fixtures: "
                + ", ".join(missing)
                + ". Disponibles: "
                + ", ".join(cols)
            )

        out = pd.DataFrame()

        for target, source in mapping.items():
            if source is not None:
                out[target] = raw[source]
            else:
                out[target] = np.nan

        out["utc_time"] = pd.to_datetime(
            out["utc_time"],
            utc=True,
            errors="coerce",
        )

        out["season"] = pd.to_numeric(
            out["season"],
            errors="coerce",
        )

        return out.dropna(
            subset=[
                "utc_time",
                "season",
                "home_team",
                "away_team",
            ]
        ).copy()

    def next_round(self):
        now = pd.Timestamp.now(tz="UTC")

        df = self.fixtures.copy()

        status = (
            df["status"]
            .fillna("")
            .astype(str)
            .str.lower()
        )

        blocked = status.str.contains(
            "finished|full-time|full time|abandoned|cancelled|canceled",
            regex=True,
        )

        future = df[
            (df["utc_time"] >= now - pd.Timedelta(hours=12))
            & (~blocked)
        ].copy()

        if future.empty:
            future = df[
                df["utc_time"] > self.history["utc_time"].max()
            ].copy()

        if future.empty:
            return pd.DataFrame()

        current_season = int(future["season"].max())

        future = future[
            future["season"] == current_season
        ].sort_values(
            ["utc_time", "match_id"]
        )

        first = future.iloc[0]

        if pd.notna(first["round"]):
            same_round = future[
                future["round"].astype(str)
                == str(first["round"])
            ].copy()

            if not same_round.empty:
                return same_round.reset_index(drop=True)

        start = first["utc_time"].floor("D")
        end = start + pd.Timedelta(days=7)

        return future[
            (future["utc_time"] >= start)
            & (future["utc_time"] < end)
        ].reset_index(drop=True)

    def _history_before(self, team, cutoff):
        return self.history[
            (self.history["team"] == team)
            & (self.history["utc_time"] < cutoff)
        ].sort_values("utc_time")

    def _league_mean(self, metric, cutoff):
        if metric not in self.history.columns:
            return np.nan

        h = self.history[
            self.history["utc_time"] < cutoff
        ]

        s = pd.to_numeric(
            h[metric],
            errors="coerce",
        )

        return float(s.mean()) if s.notna().any() else np.nan

    def _team_mean(
        self,
        team,
        metric,
        window,
        cutoff,
        season,
        venue=None,
    ):
        h = self._history_before(team, cutoff)

        if venue is not None and "venue" in h.columns:
            h = h[
                h["venue"].astype(str).str.lower() == venue
            ]

        if metric not in h.columns:
            return self._league_mean(metric, cutoff)

        if window == "last5":
            h = h.tail(5)
        elif window == "last10":
            h = h.tail(10)
        elif window == "season":
            hs = h[
                pd.to_numeric(
                    h["season"],
                    errors="coerce",
                ) == int(season)
            ]
            if not hs.empty:
                h = hs
            else:
                h = h.tail(10)

        s = pd.to_numeric(
            h[metric],
            errors="coerce",
        ).dropna()

        if s.empty:
            return self._league_mean(metric, cutoff)

        return float(s.mean())

    def _build_standard_side_row(
        self,
        cols,
        team,
        opponent,
        is_home,
        cutoff,
        season,
        expected_kind,
    ):
        row = {}

        venue_name = "home" if is_home else "away"

        for col in cols:
            if col == "attack_team":
                row[col] = team
                continue

            if col == "defence_team":
                row[col] = opponent
                continue

            if col == "is_home":
                row[col] = int(is_home)
                continue

            m = re.fullmatch(
                r"att_(last5|last10|season)_(.+)",
                col,
            )
            if m:
                row[col] = self._team_mean(
                    team,
                    m.group(2),
                    m.group(1),
                    cutoff,
                    season,
                )
                continue

            m = re.fullmatch(
                r"oppdef_(last5|last10|season)_(.+)",
                col,
            )
            if m:
                row[col] = self._team_mean(
                    opponent,
                    m.group(2),
                    m.group(1),
                    cutoff,
                    season,
                )
                continue

            m = re.fullmatch(
                r"venue_(.+)",
                col,
            )
            if m:
                row[col] = self._team_mean(
                    team,
                    m.group(1),
                    "last5",
                    cutoff,
                    season,
                    venue=venue_name,
                )
                continue

            if expected_kind == "goals":
                m = re.fullmatch(
                    r"expected_(xg|goals)_(last5|last10|season)",
                    col,
                )
                if m:
                    metric = (
                        "xg"
                        if m.group(1) == "xg"
                        else "goals"
                    )
                    attack_metric = f"{metric}_for"
                    defence_metric = f"{metric}_against"

                    row[col] = float(
                        np.nanmean(
                            [
                                self._team_mean(
                                    team,
                                    attack_metric,
                                    m.group(2),
                                    cutoff,
                                    season,
                                ),
                                self._team_mean(
                                    opponent,
                                    defence_metric,
                                    m.group(2),
                                    cutoff,
                                    season,
                                ),
                            ]
                        )
                    )
                    continue

            if expected_kind == "corners":
                m = re.fullmatch(
                    r"expected_corners_(last5|last10|season)",
                    col,
                )
                if m:
                    row[col] = float(
                        np.nanmean(
                            [
                                self._team_mean(
                                    team,
                                    "corners_for",
                                    m.group(1),
                                    cutoff,
                                    season,
                                ),
                                self._team_mean(
                                    opponent,
                                    "corners_against",
                                    m.group(1),
                                    cutoff,
                                    season,
                                ),
                            ]
                        )
                    )
                    continue

            row[col] = np.nan

        return pd.DataFrame(
            [row],
            columns=cols,
        )

    def _match_level_history(self, cutoff):
        h = self.history[
            self.history["utc_time"] < cutoff
        ].copy()

        if "venue" in h.columns:
            home = h[
                h["venue"].astype(str).str.lower() == "home"
            ].copy()
        else:
            home = h.drop_duplicates("match_id").copy()

        return home.sort_values("utc_time")

    @staticmethod
    def _recency_weights(times, cutoff, half_life=420.0):
        t = pd.to_datetime(
            times,
            utc=True,
            errors="coerce",
        )

        days = (
            cutoff - t
        ).dt.days.clip(lower=0)

        w = np.power(
            0.5,
            days.to_numpy(dtype=float) / half_life,
        )

        return np.clip(w, 0.15, 1.0)

    def _goal_baseline_lambdas(self, cutoff):
        h = self._match_level_history(cutoff)

        if h.empty:
            return 1.5, 1.2

        w = self._recency_weights(
            h["utc_time"],
            cutoff,
            420.0,
        )

        hg = pd.to_numeric(
            h["goals_for"],
            errors="coerce",
        ).fillna(0).to_numpy(dtype=float)

        ag = pd.to_numeric(
            h["goals_against"],
            errors="coerce",
        ).fillna(0).to_numpy(dtype=float)

        return (
            float(np.average(hg, weights=w)),
            float(np.average(ag, weights=w)),
        )

    def _total_baseline(self, cutoff, metric_for, metric_against, half_life):
        h = self._match_level_history(cutoff)

        if h.empty:
            return np.nan

        if (
            metric_for not in h.columns
            or metric_against not in h.columns
        ):
            return np.nan

        w = self._recency_weights(
            h["utc_time"],
            cutoff,
            half_life,
        )

        total = (
            pd.to_numeric(
                h[metric_for],
                errors="coerce",
            ).fillna(0)
            + pd.to_numeric(
                h[metric_against],
                errors="coerce",
            ).fillna(0)
        ).to_numpy(dtype=float)

        return float(
            np.average(total, weights=w)
        )

    def _latest_shrink_weight(self, table, market, fallback):
        try:
            with sqlite3.connect(DB_PATH) as conn:
                q = pd.read_sql_query(
                    f"""
                    SELECT season, shrink_weight
                    FROM {table}
                    WHERE market = ?
                      AND shrink_weight IS NOT NULL
                    ORDER BY season DESC
                    LIMIT 1
                    """,
                    conn,
                    params=(market,),
                )

            if not q.empty:
                return float(q.iloc[0]["shrink_weight"])
        except Exception:
            pass

        return float(fallback)

    def predict_goals(self, fixture):
        cutoff = fixture["utc_time"]
        season = int(fixture["season"])

        home_row = self._build_standard_side_row(
            self.goals_cols,
            fixture["home_team"],
            fixture["away_team"],
            True,
            cutoff,
            season,
            "goals",
        )

        away_row = self._build_standard_side_row(
            self.goals_cols,
            fixture["away_team"],
            fixture["home_team"],
            False,
            cutoff,
            season,
            "goals",
        )

        lh = float(
            np.clip(
                self.goals_model.predict(home_row)[0],
                0.05,
                5.5,
            )
        )

        la = float(
            np.clip(
                self.goals_model.predict(away_row)[0],
                0.05,
                5.5,
            )
        )

        raw_o25 = poisson_over(
            2.5,
            lh + la,
        )

        bh, ba = self._goal_baseline_lambdas(cutoff)
        base_o25 = poisson_over(
            2.5,
            bh + ba,
        )

        weight = self._latest_shrink_weight(
            "goals_v4_backtest_predictions",
            "over_2_5",
            0.95,
        )

        final_o25 = (
            weight * raw_o25
            + (1 - weight) * base_o25
        )

        ph, pd_, pa = result_probs(lh, la)

        return {
            "lambda_home": lh,
            "lambda_away": la,
            "over_2_5": float(np.clip(final_o25, 0, 1)),
            "home_win_raw": ph,
            "draw_raw": pd_,
            "away_win_raw": pa,
        }

    def predict_corners(self, fixture):
        cutoff = fixture["utc_time"]
        season = int(fixture["season"])

        home_row = self._build_standard_side_row(
            self.corners_cols,
            fixture["home_team"],
            fixture["away_team"],
            True,
            cutoff,
            season,
            "corners",
        )

        away_row = self._build_standard_side_row(
            self.corners_cols,
            fixture["away_team"],
            fixture["home_team"],
            False,
            cutoff,
            season,
            "corners",
        )

        lh = float(
            np.clip(
                self.corners_model.predict(home_row)[0],
                0.20,
                12.0,
            )
        )

        la = float(
            np.clip(
                self.corners_model.predict(away_row)[0],
                0.20,
                12.0,
            )
        )

        total_lambda = lh + la

        baseline = self._total_baseline(
            cutoff,
            "corners_for",
            "corners_against",
            420.0,
        )

        if pd.isna(baseline):
            baseline = total_lambda

        output = {
            "lambda_home": lh,
            "lambda_away": la,
        }

        fallbacks = {
            9.5: 0.60,
            10.5: 0.75,
            11.5: 0.65,
        }

        for line in (9.5, 10.5, 11.5):
            market = f"over_{str(line).replace('.', '_')}"

            raw = poisson_over(
                line,
                total_lambda,
            )

            base = poisson_over(
                line,
                baseline,
            )

            weight = self._latest_shrink_weight(
                "corners_v1_backtest_predictions",
                market,
                fallbacks[line],
            )

            output[market] = float(
                np.clip(
                    weight * raw
                    + (1 - weight) * base,
                    0,
                    1,
                )
            )

        return output

    def _ref_features(self, referee, cutoff):
        if self.ref_history.empty:
            return {
                "ref_matches": 0,
                "ref_total_yellow": np.nan,
                "ref_home_yellow": np.nan,
                "ref_away_yellow": np.nan,
                "ref_last10_total": np.nan,
                "ref_last20_total": np.nan,
                "ref_fouls": np.nan,
            }

        prior = self.ref_history[
            self.ref_history["utc_time"] < cutoff
        ].copy()

        if prior.empty:
            return {
                "ref_matches": 0,
                "ref_total_yellow": np.nan,
                "ref_home_yellow": np.nan,
                "ref_away_yellow": np.nan,
                "ref_last10_total": np.nan,
                "ref_last20_total": np.nan,
                "ref_fouls": np.nan,
            }

        league_total = pd.to_numeric(
            prior.get("total_yellow"),
            errors="coerce",
        ).mean()

        league_home = pd.to_numeric(
            prior.get("home_yellow"),
            errors="coerce",
        ).mean()

        league_away = pd.to_numeric(
            prior.get("away_yellow"),
            errors="coerce",
        ).mean()

        league_fouls = pd.to_numeric(
            prior.get("total_fouls"),
            errors="coerce",
        ).mean()

        if referee is None or pd.isna(referee):
            return {
                "ref_matches": 0,
                "ref_total_yellow": league_total,
                "ref_home_yellow": league_home,
                "ref_away_yellow": league_away,
                "ref_last10_total": league_total,
                "ref_last20_total": league_total,
                "ref_fouls": league_fouls,
            }

        h = prior[
            prior["referee"].astype(str)
            == str(referee)
        ].sort_values("utc_time")

        if h.empty:
            return {
                "ref_matches": 0,
                "ref_total_yellow": league_total,
                "ref_home_yellow": league_home,
                "ref_away_yellow": league_away,
                "ref_last10_total": league_total,
                "ref_last20_total": league_total,
                "ref_fouls": league_fouls,
            }

        strength = 20.0

        def shrunk(col, league):
            if col not in h.columns:
                return league

            s = pd.to_numeric(
                h[col],
                errors="coerce",
            ).dropna()

            if s.empty:
                return league

            return float(
                (
                    s.sum()
                    + strength * league
                )
                / (len(s) + strength)
            )

        total = pd.to_numeric(
            h["total_yellow"],
            errors="coerce",
        ).dropna()

        fouls = pd.to_numeric(
            h["total_fouls"],
            errors="coerce",
        ).dropna()

        return {
            "ref_matches": len(h),
            "ref_total_yellow": shrunk(
                "total_yellow",
                league_total,
            ),
            "ref_home_yellow": shrunk(
                "home_yellow",
                league_home,
            ),
            "ref_away_yellow": shrunk(
                "away_yellow",
                league_away,
            ),
            "ref_last10_total": float(
                total.tail(10).mean()
            ) if not total.empty else league_total,
            "ref_last20_total": float(
                total.tail(20).mean()
            ) if not total.empty else league_total,
            "ref_fouls": float(
                fouls.mean()
            ) if not fouls.empty else league_fouls,
        }

    def _league_cards_features(self, cutoff, season, side):
        h = self._match_level_history(cutoff)

        if h.empty:
            return {
                "league_last50_cards": np.nan,
                "league_season_cards": np.nan,
                "league_side_cards": np.nan,
            }

        total_cards = (
            pd.to_numeric(
                h["yellow_for"],
                errors="coerce",
            )
            + pd.to_numeric(
                h["yellow_against"],
                errors="coerce",
            )
        )

        last50 = float(
            total_cards.tail(50).mean()
        )

        hs = h[
            pd.to_numeric(
                h["season"],
                errors="coerce",
            ) == int(season)
        ]

        if hs.empty:
            season_avg = last50
        else:
            season_avg = float(
                (
                    pd.to_numeric(
                        hs["yellow_for"],
                        errors="coerce",
                    )
                    + pd.to_numeric(
                        hs["yellow_against"],
                        errors="coerce",
                    )
                ).mean()
            )

        if side == "home":
            side_avg = float(
                pd.to_numeric(
                    h["yellow_for"],
                    errors="coerce",
                ).tail(50).mean()
            )
        else:
            side_avg = float(
                pd.to_numeric(
                    h["yellow_against"],
                    errors="coerce",
                ).tail(50).mean()
            )

        return {
            "league_last50_cards": last50,
            "league_season_cards": season_avg,
            "league_side_cards": side_avg,
        }

    def _build_cards_row(
        self,
        team,
        opponent,
        side,
        referee,
        cutoff,
        season,
    ):
        ref = self._ref_features(
            referee,
            cutoff,
        )

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
                    referee
                    if referee is not None and not pd.isna(referee)
                    else np.nan
                )
                continue

            if col in ref:
                row[col] = ref[col]
                continue

            if col in league:
                row[col] = league[col]
                continue

            if col == "ref_side_yellow":
                row[col] = (
                    ref["ref_home_yellow"]
                    if side == "home"
                    else ref["ref_away_yellow"]
                )
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

                ref_y = (
                    ref["ref_home_yellow"]
                    if side == "home"
                    else ref["ref_away_yellow"]
                )

                values = [
                    v for v in (team_y, opp_y, ref_y)
                    if not pd.isna(v)
                ]

                row[col] = (
                    float(np.mean(values))
                    if values
                    else np.nan
                )
                continue

            row[col] = np.nan

        return pd.DataFrame(
            [row],
            columns=self.cards_cols,
        )

    def predict_cards(self, fixture):
        cutoff = fixture["utc_time"]
        season = int(fixture["season"])
        referee = fixture.get("referee", np.nan)

        home_row = self._build_cards_row(
            fixture["home_team"],
            fixture["away_team"],
            "home",
            referee,
            cutoff,
            season,
        )

        away_row = self._build_cards_row(
            fixture["away_team"],
            fixture["home_team"],
            "away",
            referee,
            cutoff,
            season,
        )

        mu_home = float(
            np.clip(
                self.cards_model.predict(home_row)[0],
                0.05,
                5.0,
            )
        )

        mu_away = float(
            np.clip(
                self.cards_model.predict(away_row)[0],
                0.05,
                5.0,
            )
        )

        mu_total = mu_home + mu_away

        match_hist = self._match_level_history(cutoff)

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

        output = {
            "mu_home": mu_home,
            "mu_away": mu_away,
            "nb_alpha": alpha,
        }

        fallbacks = {
            3.5: 0.80,
            4.5: 0.85,
            5.5: 1.00,
        }

        for line in (3.5, 4.5, 5.5):
            market = f"over_{str(line).replace('.', '_')}"

            raw = nb_over(
                line,
                mu_total,
                alpha,
            )

            base = nb_over(
                line,
                baseline,
                alpha,
            )

            weight = self._latest_shrink_weight(
                "cards_v2_backtest_predictions",
                market,
                fallbacks[line],
            )

            output[market] = float(
                np.clip(
                    weight * raw
                    + (1 - weight) * base,
                    0,
                    1,
                )
            )

        return output

    def predict_fixture(self, fixture):
        goals = self.predict_goals(fixture)
        corners = self.predict_corners(fixture)
        cards = self.predict_cards(fixture)

        return {
            "match_id": int(fixture["match_id"]),
            "season": int(fixture["season"]),
            "round": fixture.get("round"),
            "utc_time": fixture["utc_time"],
            "home_team": fixture["home_team"],
            "away_team": fixture["away_team"],
            "referee": fixture.get("referee"),
            "goals": goals,
            "corners": corners,
            "cards": cards,
        }

    def predict_next_round(self):
        fixtures = self.next_round()

        if fixtures.empty:
            return []

        predictions = []

        for _, fixture in fixtures.iterrows():
            predictions.append(
                self.predict_fixture(fixture)
            )

        return predictions
