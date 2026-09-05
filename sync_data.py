from api_football import get_available_seasons, get_season_fixtures
from config import DATA_START_SEASON
from database import init_db, log_sync, upsert_matches


def sync_matches_from_2023(progress_callback=None):
    """
    Primera sincronización: descarga fixtures/resultados desde 2023.
    Las estadísticas detalladas por partido se añadirán en el siguiente módulo,
    para no gastar la cuota de API innecesariamente durante el arranque.
    """
    init_db()

    seasons = [
        s for s in get_available_seasons()
        if s >= DATA_START_SEASON
    ]

    total = 0

    for idx, season in enumerate(seasons, start=1):
        fixtures = get_season_fixtures(season)
        saved = upsert_matches(fixtures)
        total += saved
        log_sync("season_fixtures", season, f"{saved} fixtures guardados")

        if progress_callback:
            progress_callback(
                idx / max(len(seasons), 1),
                f"Temporada {season}: {saved} partidos guardados"
            )

    return total, seasons
