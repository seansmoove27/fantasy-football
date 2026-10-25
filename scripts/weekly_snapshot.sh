#!/bin/sh
# Refresh data, rebuild both models, and save this week's lines and boards to snapshots/ so live picks can be
# scored later against what we actually predicted. Run Friday and again Sunday morning.
# Usage: sh scripts/weekly_snapshot.sh <season> <week>   (expects inputs/<season>_week<week>_wind.csv)
set -e
SEASON=$1 WEEK=$2
cd "$(dirname "$0")/.."
WIND=inputs/${SEASON}_week${WEEK}_wind.csv
STAMP=$(date +%Y%m%d_%H%M)
mkdir -p snapshots
OUT=snapshots/${SEASON}_w${WEEK}_${STAMP}
uv run python scripts/download_data.py "$SEASON" "$SEASON"
uv run python -I dst/dst_v2.py build data/raw
uv run python -I kicker/k_model.py build data/raw
uv run python -I dst/dst_v2.py predict data/raw "$SEASON" "$WEEK" "$WIND" > "${OUT}_dst.txt"
uv run python -I kicker/k_model.py predict data/raw "$SEASON" "$WEEK" "$WIND" > "${OUT}_k.txt"
uv run python -I -c "
import sys, pandas as pd
g = pd.read_csv('data/raw/games.csv')
g[(g.season == $SEASON) & (g.week == $WEEK)][['game_id', 'gameday', 'away_team', 'home_team', 'spread_line',
    'total_line', 'roof', 'away_qb_name', 'home_qb_name']].to_csv('${OUT}_lines.csv', index=False)
"
echo "saved ${OUT}_{dst,k}.txt and ${OUT}_lines.csv"
