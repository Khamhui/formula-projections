# F1 Prediction Engine

AI-powered Formula 1 race prediction system using 75 years of historical data, multi-dimensional ELO ratings, Monte Carlo simulation, and XGBoost machine learning. 150-feature model with 3.0 position MAE, 92% podium accuracy, and 97% winner accuracy.

Inspired by [theGreenCoding's tennis prediction model](https://github.com/theGreenCoding) that achieved 85% accuracy at the Australian Open.

## What It Predicts

- **Race winner** and full finishing order (XGBoost + stacking ensemble)
- **Win / podium / points probabilities** via 10,000 Monte Carlo simulations
- **Expected points** and position confidence ranges (25th–75th percentile)
- **DNF risk** per driver (circuit-type aware)
- **Head-to-head** matchups between any two drivers
- **ELO power rankings** across 6 dimensions with trend sparklines

## How It Works

### Data Pipeline

```
Ingest (3 sources) → Feature Engineering (150 features) → XGBoost Train → Predict → Monte Carlo Sim → Dashboard
```

### Data Sources

| Source | Coverage | Data |
|--------|----------|------|
| **Jolpica-F1** | 1950–present | Race results, qualifying, standings, circuits, pit stops, lap times |
| **FastF1** | 2018+ | Granular lap data, tire compounds, sector times, speed traps, weather, telemetry |
| **OpenF1** | 2023+ | Real-time telemetry (3.7Hz), overtakes, intervals, live positions |

### ELO Rating System

Adapted from chess for multi-player motorsport — each race generates (n*(n-1))/2 pairwise comparisons:

| Dimension | K-factor | Description |
|-----------|----------|-------------|
| **Overall** | 6 | General race performance |
| **Circuit-type** | 6 | Street / high-speed / technical / mixed specialization |
| **Qualifying** | 4 | One-lap pace (lower K — less variance) |
| **Wet-weather** | 8 | Performance in rain (higher K — reveals more delta) |
| **Constructor** | 4 | Team/car performance |
| **Teammate H2H** | 6 | Intra-team head-to-head |

ELO ratings are rebuilt chronologically from history and updated after each race/qualifying session.

### Feature Matrix (150 features per driver per race)

**130 base features:**

- ELO ratings and differentials (6 dimensions: overall, circuit-type, qualifying, wet, constructor, teammate H2H)
- Rolling form windows (last 3/5/10/20 races)
- Circuit-specific performance (EWM position, positions gained, win streak, qualifying history at circuit)
- Circuit DNA (grid-position correlation, front row win rate, attrition rate, position variance)
- Grid × circuit interactions (grid weighted by correlation, importance score, expected finish)
- Constructor-at-circuit history (average, recent trend, race count)
- Cumulative championship standings (points to leader, WDC/WCC position per round)
- Compromised finish detection (z-score based, flags races where driver lost abnormally many positions)
- Grid position and qualifying pace
- Season momentum (EWM, form vs season average)
- Tire degradation and strategy (2018+)
- Weather conditions (auto-detected from FastF1)
- Constructor strength and development trajectory
- DNF risk indicators (mechanical reliability, crash history, DNF streak)
- Teammate comparison (ELO diff, qualifying H2H win rate, positions gained delta)
- Pit stop efficiency (team average, clean stop rate)

**20 FastF1 features (2018+):**

- Practice session pace (FP1/FP2/FP3 gap to leader)
- FP2 long run analysis (race pace indicator, tire degradation rate)
- Track status performance (SC/VSC gains)
- Sector times and speed trap data

### Model

**XGBoost stacking ensemble** with calibrated auxiliary classifiers:
- Position predictor (regression)
- Win classifier (binary)
- Podium classifier (binary)
- Points classifier (binary)
- DNF classifier (binary)

Trained with time-series cross-validation to prevent data leakage. Features are extracted at the state *before* each race (no future information).

**Current metrics (9,467-row dataset, 1950–2025):**

| Metric | Value |
|--------|-------|
| Position MAE | 3.019 |
| Podium accuracy | 92.2% |
| Winner accuracy | 96.6% |

### Monte Carlo Simulation

10,000 vectorized simulations (NumPy) per race weekend:
- Position noise with correlated team effects
- DNF sampling using circuit-type-aware probabilities
- Multi-car incidents (Gaussian-weighted grid position sampling, circuit-type-aware rates)
- Pit strategy variance and safety car probability by circuit type
- Wet conditions modifier
- Outputs: win%, podium%, points%, expected points, median position, IQR range

## Quick Start

### Double-click launchers (macOS)

The easiest way — no terminal needed:

- **`web-dashboard.command`** — Opens the web dashboard in your browser
- **`dashboard.command`** — Opens the terminal (Rich) dashboard

### Command line

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run full pipeline (ingest → features → train)
python -m data.pipeline --step all

# Predict next weekend
python -m data.predict_weekend

# Launch web dashboard
python -m src.app                 # http://localhost:5050
python -m src.app --port 8080     # Custom port
```

## Web Dashboard

Terminal-aesthetic single-page dashboard with 4-column layout:

| Column | Content |
|--------|---------|
| **Predictions** | Stat pills (predicted winner, win%, gap to P2), full 20-driver prediction grid, gain/loss insights |
| **Session Results** | Qualifying (Q1/Q2/Q3), sprint race, race result with grid-to-finish delta (+/-) |
| **Standings + ELO** | Driver championship, constructor standings (with ELO), ELO power rankings with SVG sparklines |
| **Charts + News** | Win probability, expected points, podium probability, DNF risk (horizontal bar charts), RSS news feed |

Features:
- OLED black dark mode / pure white light mode (toggle + localStorage persistence)
- Circuit type badges (street / high-speed / technical / mixed)
- Team color bars throughout
- Race/round selector for historical predictions
- Zero JavaScript dependencies — pure CSS charts, SVG sparklines, vanilla JS for theme toggle and RSS fetch

## Terminal Dashboard

Rich-based terminal UI (alternative to web):

```bash
python -m data.dashboard
```

## Project Structure

```
├── data/
│   ├── ingest/                  # Data source clients
│   │   ├── jolpica.py           # Jolpica-F1 API (1950–present)
│   │   ├── fastf1_ingest.py     # FastF1 library (2018+)
│   │   ├── openf1_client.py     # OpenF1 real-time API (2023+)
│   │   └── openf1_penalties.py  # Penalty data extraction
│   ├── features/                # Feature engineering
│   │   ├── elo.py               # Multi-dimensional ELO system
│   │   └── engineer.py          # Full feature matrix builder (150 features)
│   ├── models/                  # ML models
│   │   ├── predictor.py         # XGBoost stacking ensemble
│   │   ├── simulator.py         # 10k Monte Carlo simulation
│   │   ├── tuner.py             # Hyperparameter tuning
│   │   ├── backtest.py          # Historical backtesting
│   │   └── explain.py           # SHAP / feature importance
│   ├── pipeline.py              # End-to-end pipeline orchestrator
│   ├── predict_weekend.py       # Predict next race weekend
│   ├── dashboard.py             # Rich terminal dashboard
│   └── cache/                   # Cached data (gitignored)
│       └── processed/           # Parquet files + prediction CSVs
├── src/                         # Flask web dashboard
│   ├── app.py                   # Routes, data loading, template context
│   ├── shared.py                # Constants (drivers, teams, colors, helpers)
│   ├── templates/
│   │   └── terminal.html        # Single-page Jinja2 template
│   └── static/
│       └── style.css            # Terminal-aesthetic CSS
├── dashboard.command             # macOS launcher (terminal dashboard)
├── web-dashboard.command         # macOS launcher (web dashboard)
├── requirements.txt
└── .env.example
```

## Roadmap

- [x] Jolpica historical data ingestion (1950–present)
- [x] FastF1 granular data ingestion (2018+)
- [x] OpenF1 real-time client (2023+)
- [x] Multi-dimensional ELO rating system (6 dimensions)
- [x] Feature engineering pipeline (150 features)
- [x] XGBoost stacking ensemble (position, podium, winner, points, DNF)
- [x] Monte Carlo simulation (10k sims, vectorized, multi-car incidents)
- [x] Web dashboard (Flask, 4-column terminal aesthetic)
- [x] Terminal dashboard (Rich)
- [x] macOS double-click launchers
- [x] Wet race detection (auto-detected from FastF1 weather data)
- [x] Circuit DNA features (grid correlation, attrition, front row rates)
- [x] Circuit-specific driver/constructor history (EWM, streaks, qualifying)
- [x] Compromised finish detection (z-score based)
- [x] Cumulative per-round championship standings
- [x] Betting odds integration and value detection
- [ ] Real-time race predictions during qualifying/race
- [ ] Championship probability simulations (full season Monte Carlo)
