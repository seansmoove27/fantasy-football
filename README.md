# fantasy-football

Weekly D/ST and kicker streaming models for the **Limers** ESPN league, backtested on nflverse play-by-play (2015–2026) under the league's exact scoring.

## League scoring

- **D/ST:** sack 1 · INT / fumble recovery / safety / blocked kick 2 · any return TD 6
  - Points allowed: 0 → +5 · 1–6 → +4 · 7–13 → +3 · 14–17 → +1 · 18–27 → 0 · 28–34 → −1 · 35–45 → −3 · 46+ → −5
  - Yards allowed: <100 → +5 · 100–199 → +3 · 200–299 → +2 · 300–349 → 0 · 350–399 → −1 · 400–449 → −3 · 450–499 → −5 · 500–549 → −6 · 550+ → −7
- **Kicker:** PAT 1 · FG 0–39 3 · 40–49 4 · 50–59 5 · 60+ 6 · missed FG −1

## Findings

| | Weekly rank correlation (2019–2026 walk-forward) |
|---|---|
| FantasyPros weekly DST ECR | 0.318 |
| D/ST v1: opponent implied total + spread | 0.348 |
| **D/ST v2: ridge on lines, form, wind** | **0.355** |

- v2 beats FantasyPros ECR by +0.037 (95% CI +0.021 to +0.053) over 101 weeks, and beats v1 in 8 of 8 seasons.
- **Caveat:** historical lines are closing lines; FantasyPros snapshots are Fridays. Re-pull lines Sunday morning to get the same information live.
- About 85% of D/ST scoring is luck (R² ≈ 0.15). A defense's own sacks and takeaways barely predict anything; the opponent and the betting line do.
- Kickers are about 4% predictable. Team implied total is the best signal; red-zone "stall rate" is useless.

## Layout

| Path | What it does |
|---|---|
| `scripts/download_data.py` | Downloads nflverse pbp, schedules + lines, and the FantasyPros ECR archive into `data/raw/` |
| `dst/dst_v2.py` | Current D/ST model: `build`, `eval` (walk-forward), `predict` (weekly board) |
| `dst/compare_fp.py` | Compares walk-forward predictions with FantasyPros weekly ECR |
| `dst/dst_v1_backtest.py` | First version: single-indicator tests, implied-total model |
| `kicker/k_backtest.py` | Kicker indicator backtest |
| `kicker/kicker-analysis-local.patch` | Local edits used to run [tdub29/Kicker-Analysis](https://github.com/tdub29/Kicker-Analysis) (missed FG = −1, output path) |
| `audit/` | Independent checks of the v1 backtests (spread sign, leakage, scoring parity) |
| `exploration/` | One-off scripts from the initial research |
| `inputs/` | Weekly hand inputs, e.g. wind forecasts for outdoor games |

## Weekly run

```sh
uv run python scripts/download_data.py 2026 2026            # refresh this season + lines
uv run python -I dst/dst_v2.py build data/raw                # rebuild features
uv run python -I dst/dst_v2.py predict data/raw 2026 7 inputs/2026_week7_wind.csv
```

`inputs/<season>_week<N>_wind.csv` has columns `team,wind`: the home team of each outdoor game and its forecast wind in mph.

Model check (optional):

```sh
uv run python -I dst/dst_v2.py eval data/raw
uv run python -I dst/compare_fp.py data/raw/db_fpecr.parquet data/raw/games.csv data/dst_v2_walkforward.parquet
```

## Data sources

- [nflverse-data](https://github.com/nflverse/nflverse-data) play-by-play and [nfldata](https://github.com/nflverse/nfldata) schedules
- [DynastyProcess data](https://github.com/dynastyprocess/data) FantasyPros ECR archive
