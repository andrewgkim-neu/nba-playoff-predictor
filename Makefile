.PHONY: help install init-db fetch-games fetch-boxscores fetch-playoff-series build-features build-player-stats build-player-features build-player-profiles load-odds pipeline status evaluate train simulate gui clean-cache clean-db clean-features

help:
	@echo "Targets:"
	@echo "  install               Install Python dependencies"
	@echo "  init-db               Create the SQLite database schema"
	@echo "  fetch-games           Pull game list for all configured seasons"
	@echo "  fetch-boxscores       Pull team box scores for games missing them"
	@echo "  fetch-playoff-series  Backfill round/series/game-num into games"
	@echo "  build-features        Build the game_features analytics table"
	@echo "  build-player-stats    Extract player box scores from cache -> player_game_stats"
	@echo "  build-player-features Add lineup-adjusted team ratings to game_features"
	@echo "  build-player-profiles Aggregate per-player season profiles (RS / playoffs split)"
	@echo "  load-odds             Load data/odds.csv into the odds table"
	@echo "  pipeline              Run all of the above in order (no odds)"
	@echo "  status                Print row counts and coverage by season"
	@echo "  evaluate              Walk-forward validation report for the models"
	@echo "  train                 Fit the final model on all data -> data/model.pkl"
	@echo "  compare               Score the model against sportsbook odds"
	@echo "  edge                  Test whether the market line as a feature beats the line alone"
	@echo "  simulate              Simulate a series, e.g. make simulate HOME=BOS AWAY=DAL"
	@echo "  gui                   Launch the snake-draft browser game (Streamlit)"
	@echo "  clean-cache           Delete cached raw JSON (forces re-fetch)"
	@echo "  clean-db              Delete the SQLite database"
	@echo "  clean-features        Drop only the game_features table"

install:
	pip install -r requirements.txt

init-db:
	python -m src.db

fetch-games:
	python -m src.fetch_games

fetch-boxscores:
	python -m src.fetch_boxscores

fetch-playoff-series:
	python -m src.fetch_playoff_series

build-features:
	python -m src.build_features

build-player-stats:
	python -m src.build_player_stats

build-player-features:
	python -m src.build_player_features

build-player-profiles:
	python -m src.build_player_profiles

gui:
	streamlit run app.py

load-odds:
	python -m src.load_odds

pipeline: init-db fetch-games fetch-boxscores fetch-playoff-series build-features build-player-stats build-player-features build-player-profiles

status:
	python -m src.db status

evaluate:
	python -m src.model evaluate

train:
	python -m src.model train

compare:
	python -m src.model compare

edge:
	python -m src.model edge

# Usage: make simulate HOME=BOS AWAY=DAL [SEASON=2024-25] [SIMS=20000]
SEASON ?=
SIMS ?= 20000
simulate:
	python -m src.model simulate --home $(HOME) --away $(AWAY) $(if $(SEASON),--season $(SEASON),) --sims $(SIMS)

clean-cache:
	rm -rf data/raw/*
	touch data/raw/.gitkeep

clean-db:
	rm -f data/nba.db data/nba.db-wal data/nba.db-shm

clean-features:
	sqlite3 data/nba.db "DROP TABLE IF EXISTS game_features"
