# ⚽ Football Edge Pro

Football analytics dashboard built with Python + Streamlit.

## Features

- 🇪🇸 LaLiga and 🇳🇴 Eliteserien engines kept independent
- 🔎 Jornada Scanner
- ⚽ Goals models
- 🚩 Corners models
- 🟨 Cards V3 with referee profile
- 🎯 Shots on target, total and by team
- 💶 Fair odds and minimum odds for configurable EV
- 📊 1X2 probability / fair-price comparison
- 📱 Mobile-friendly dropdown navigation
- 🌙 Premium dark sports dashboard

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Files needed for the live app

The repository should include the trained model files in `models/` and the current SQLite databases in `data/`.

Keep local caches and backups out of GitHub. The included `.gitignore` already excludes them.

## Important

Model probability is not the same as guaranteed profitability. Markets should only be treated as betting candidates when the price produces sufficient expected value and the market has been validated appropriately.

## Main app

`streamlit_app.py`
