"""Download the raw data the models read into data/raw/.

  - nflverse play-by-play, one parquet per season
  - nflverse schedules with closing spread/total, roof, wind (games.csv)
  - DynastyProcess archive of FantasyPros ECR snapshots (for compare_fp.py) and player ID crosswalk
  - nflverse weekly player stats and snap counts, ffverse expected fantasy points (players/ models)

Usage: python scripts/download_data.py [first_season] [last_season]
Re-run each week to refresh the current season's play-by-play and lines.
"""
import sys
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / 'data' / 'raw'
PBP = 'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{}.parquet'
GAMES = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
FPECR = 'https://github.com/dynastyprocess/data/raw/master/files/db_fpecr.parquet'
PLAYERIDS = 'https://github.com/dynastyprocess/data/raw/master/files/db_playerids.csv'
STATS = 'https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{}.parquet'
SNAPS = 'https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{}.parquet'
EP = 'https://github.com/ffverse/ffopportunity/releases/download/latest-data/ep_weekly_{}.parquet'

first, last = (int(a) for a in sys.argv[1:3]) if len(sys.argv) == 3 else (2015, 2026)
RAW.mkdir(parents=True, exist_ok=True)
targets = [(PBP.format(y), f'pbp_{y}.parquet') for y in range(first, last + 1)]
for url, name in [(STATS, 'stats_player_week_{}.parquet'), (SNAPS, 'snap_counts_{}.parquet'), (EP, 'ep_weekly_{}.parquet')]:
    targets += [(url.format(y), name.format(y)) for y in range(max(first, 2018), last + 1)]
targets += [(GAMES, 'games.csv'), (FPECR, 'db_fpecr.parquet'), (PLAYERIDS, 'db_playerids.csv')]
for url, name in targets:
    print('downloading', name)
    urllib.request.urlretrieve(url, RAW / name)
