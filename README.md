# F1 Prediction Engine

AI-powered Formula 1 race prediction system using 75 years of historical data, multi-dimensional ELO ratings, and XGBoost machine learning.

Inspired by [theGreenCoding's tennis prediction model](https://github.com/theGreenCoding) that achieved 85% accuracy at the Australian Open.

## What It Predicts

- **Race winner** and full finishing order
- **Podium** probabilities for each driver
- **Points finish** likelihood
- **Head-to-head** matchups between any two drivers
- **World championship** trajectory

## How It Works

### Data Sources (layered)

| Source | Coverage | Data |
|--------|----------|------|
| **Jolpica-F1** | 1950–present | Race results, qualifying, standings, circuits, pit stops, lap times |
| **FastF1** | 2018+ | Granular lap data, tire compounds, sector times, speed traps, weather, telemetry |
| **OpenF1** | 2023+ | Real-time telemetry (3.7Hz), overtakes, intervals, live positions |

### ELO Rating System

Adapted from chess for multi-player motorsport:

- **Overall ELO** — general race performance
- **Circuit-type ELO** — street / high-speed / technical / mixed specialization
- **Qualifying ELO** — one-lap pace
- **Wet-weather ELO** — performance in rain
- **Constructor ELO** — team/car performance
- **Teammate H2H** — intra-team comparison

### Feature Matrix (81+ features per driver per race)

- ELO ratings and differentials
- Rolling form (last 3/5/10/20 races)
- Circuit-specific performance history
- Grid position and qualifying pace
- Season momentum and trend
- Tire degradation and strategy (2018+)
- Weather conditions
- Constructor strength

### Model

**XGBoost** — same algorithm that powered the 85% tennis predictions. Trained with time-series cross-validation to prevent data leakage.

## Quick Start

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run full pipeline (ingest → features → train)
python -m data.pipeline --step all

# Or step by step
python -m data.pipeline --step ingest      # ~15 min first time
python -m data.pipeline --step features    # ~2 min
python -m data.pipeline --step train       # ~1 min
```

## Project Structure

```
├── data/
│   ├── ingest/           # Data source clients
│   │   ├── jolpica.py    # Jolpica-F1 API (1950–present)
│   │   ├── fastf1_ingest.py  # FastF1 library (2018+)
│   │   └── openf1_client.py  # OpenF1 real-time API (2023+)
│   ├── features/         # Feature engineering
│   │   ├── elo.py        # Multi-dimensional ELO system
│   │   └── engineer.py   # Full feature matrix builder
│   ├── models/           # ML models
│   │   └── predictor.py  # XGBoost predictor (position, podium, winner)
│   ├── notebooks/        # Jupyter exploration
│   │   └── 01_explore_and_train.ipynb
│   ├── pipeline.py       # End-to-end pipeline
│   └── cache/            # Cached data (gitignored)
├── src/                  # Next.js dashboard (coming soon)
├── requirements.txt
└── .env.example
```

## Roadmap

- [x] Jolpica historical data ingestion (1950–present)
- [x] FastF1 granular data ingestion (2018+)
- [x] OpenF1 real-time client (2023+)
- [x] Multi-dimensional ELO rating system
- [x] Feature engineering pipeline (81+ features)
- [x] XGBoost prediction models (position, podium, winner, points)
- [ ] Wet race detection (automated from weather data)
- [ ] Live dashboard (Next.js)
- [ ] Real-time race predictions during qualifying/race
- [ ] Championship probability simulations (Monte Carlo)
- [ ] Betting odds integration and value detection
- [ ] Multi-sport expansion (tennis, football)
