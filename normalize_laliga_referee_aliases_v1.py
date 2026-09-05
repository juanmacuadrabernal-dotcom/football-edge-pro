from __future__ import annotations

import re
import sqlite3
import subprocess
import sys
import unicodedata
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "laliga.db"
FEATURE_SCRIPT = BASE_DIR / "build_laliga_referee_features_v1.py"


def norm_text(value):
    if value is None:
        return ""

    s = str(value).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(
        ch for ch in s
        if not unicodedata.combining(ch)
    )
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokens(value):
    return tuple(norm_text(value).split())


def token_set(value):
    return set(tokens(value))


def choose_canonical(names):
    """
    Une automáticamente un nombre corto con un nombre más completo cuando:
    - comparten el primer nombre,
    - todos los tokens del corto aparecen en el largo,
    - existe un único candidato razonable.

    Ejemplo:
    Javier Alberola -> Javier Alberola Rojas
    """
    unique_names = sorted(
        {str(x).strip() for x in names if pd.notna(x) and str(x).strip()},
        key=lambda x: (len(tokens(x)), len(x)),
    )

    mapping = {name: name for name in unique_names}
    ambiguous = []

    # Preferimos nombres más largos como canónicos.
    for short in unique_names:
        st = tokens(short)
        ss = set(st)

        if len(st) < 2:
            continue

        candidates = []

        for long_name in unique_names:
            if long_name == short:
                continue

            lt = tokens(long_name)
            ls = set(lt)

            if len(lt) <= len(st):
                continue

            if st[0] != lt[0]:
                continue

            if ss.issubset(ls):
                candidates.append(long_name)

        if len(candidates) == 1:
            mapping[short] = candidates[0]

        elif len(candidates) > 1:
            # Elegimos solo si uno es claramente el más completo y
            # los demás son también subconjuntos suyos.
            candidates = sorted(
                candidates,
                key=lambda x: (len(tokens(x)), len(x)),
                reverse=True,
            )

            best = candidates[0]
            best_set = token_set(best)

            if all(
                token_set(c).issubset(best_set)
                for c in candidates[1:]
            ):
                mapping[short] = best
            else:
                ambiguous.append((short, candidates))

    # Resolver cadenas A -> B -> C.
    changed = True

    while changed:
        changed = False

        for k, v in list(mapping.items()):
            vv = mapping.get(v, v)

            if vv != v:
                mapping[k] = vv
                changed = True

    return mapping, ambiguous


def main():
    print()
    print("=" * 92)
    print("LALIGA EDGE PRO - NORMALIZACION DE NOMBRES DE ARBITROS")
    print("=" * 92)

    if not DB_PATH.exists():
        raise SystemExit(f"No existe: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            """
            SELECT *
            FROM referee_match_history
            ORDER BY utc_time, match_id
            """,
            conn,
        )

    if df.empty:
        raise SystemExit("referee_match_history está vacía.")

    if "referee" not in df.columns:
        raise SystemExit("No existe la columna referee.")

    before = int(df["referee"].nunique(dropna=True))

    mapping, ambiguous = choose_canonical(
        df["referee"].dropna().tolist()
    )

    changed_map = {
        k: v
        for k, v in mapping.items()
        if k != v
    }

    print(f"Árbitros distintos ANTES: {before}")
    print(f"Aliases encontrados automáticamente: {len(changed_map)}")
    print()

    if changed_map:
        print("ALIASES QUE SE VAN A UNIFICAR:")
        print("-" * 92)

        for old, new in sorted(changed_map.items()):
            print(f"{old:<40} -> {new}")

    if ambiguous:
        print()
        print("CASOS AMBIGUOS NO TOCADOS:")
        print("-" * 92)

        for old, candidates in ambiguous:
            print(f"{old}: {', '.join(candidates)}")

    # Conservamos el nombre bruto por seguridad.
    if "referee_raw" not in df.columns:
        df["referee_raw"] = df["referee"]

    df["referee"] = df["referee"].map(
        lambda x: mapping.get(str(x).strip(), str(x).strip())
        if pd.notna(x)
        else x
    )

    after = int(df["referee"].nunique(dropna=True))

    print()
    print(f"Árbitros distintos DESPUÉS: {after}")
    print(f"Reducción de aliases: {before - after}")

    # Comprobamos que ningún partido se pierda.
    if df["match_id"].duplicated().any():
        raise RuntimeError(
            "Se detectaron match_id duplicados inesperadamente. "
            "No se guardará nada."
        )

    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql(
            "referee_match_history",
            conn,
            if_exists="replace",
            index=False,
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_referee_match_history_match
            ON referee_match_history(match_id)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_referee_match_history_ref_time
            ON referee_match_history(referee, utc_time)
            """
        )

        conn.commit()

    print()
    print("Histórico normalizado guardado correctamente.")

    if not FEATURE_SCRIPT.exists():
        print()
        print(
            "No encuentro build_laliga_referee_features_v1.py. "
            "La normalización sí se ha guardado, pero tendrás que "
            "reconstruir las features manualmente."
        )
        return

    print()
    print("=" * 92)
    print("RECONSTRUYENDO FEATURES PRE-PARTIDO CON LOS NOMBRES YA UNIFICADOS")
    print("=" * 92)
    print()

    result = subprocess.run(
        [sys.executable, str(FEATURE_SCRIPT)],
        cwd=str(BASE_DIR),
    )

    if result.returncode != 0:
        raise SystemExit(
            f"El histórico quedó normalizado, pero el constructor de "
            f"features terminó con código {result.returncode}."
        )

    print()
    print("=" * 92)
    print("NORMALIZACION + REBUILD COMPLETADOS")
    print("=" * 92)
    print(
        "Siguiente paso: Cards V3 con variables del árbitro y "
        "comparación directa contra Cards V2."
    )
    print("=" * 92)


if __name__ == "__main__":
    main()
