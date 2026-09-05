from pathlib import Path

APP_NAME = "Eliteserien Edge Pro"
COUNTRY = "Norway"
LEAGUE_NAME = "Eliteserien"
DATA_START_SEASON = 2023

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "eliteserien.db"

API_BASE_URL = "https://v3.football.api-sports.io"
TIMEZONE = "Europe/Oslo"
