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
- Kickers are about 4% predictable. Red-zone "stall rate" is useless.
- **Kicker model** (`kicker/k_model.py`): ridge on implied total, spread, dome, wind and recent team kicker scoring/opponent kicker points allowed. Walk-forward 2019–2026 weekly rank correlation 0.185 vs 0.148 for implied total alone (+0.037, 95% CI +0.022 to +0.053); top-5 picks beat the weekly average by 1.10 pts vs 0.94.

- **QB/RB/WR/TE (`players/player_model.py`): FantasyPros weekly ECR is not beaten.** Walk-forward 2023–2026 (51 weeks), top-N players who played. On close start/sit pairs (ECR ranks within 10), ECR picks the higher scorer 55–57% of the time; a gradient-boosted model on usage, expected fantasy points, Vegas and matchup loses 0.5–2.6 points of that, a blend toward it picks weight ≈0, and a small ridge on signals ECR consistently misses (xFP above ECR for QBs, favorites for RBs, wind for WRs, TD luck for TEs) ties at best. The residual model does give better-calibrated point totals (RB MAE 5.57 vs 5.65), useful for win-probability simulation, not for ranking.

## Accuracy experiments (Oct 2026)

Protocol (`experiments/harness.py`): ideas are judged on 2019–2024 walk-forward only and accepted when the weekly
rank-correlation gain over the current model has a 95% CI above zero; 2025–2026 is held out for one final check.

| Idea | Model | Gain vs current (95% CI) | Result |
|---|---|---|---|
| Coach's 4th-down kick rate in long-FG range | K | −0.003 (−0.007, +0.002) | rejected |
| Kicker's own form, FG%, 50+ attempt rate | K | −0.003 (−0.015, +0.010) | rejected |
| Game temperature | K | +0.003 (−0.001, +0.007) | rejected (close) |
| Opposing starting QB: sack rate, INT rate, experience, new starter | D/ST | +0.003 (−0.003, +0.008) | rejected |
| Model PA / YA / sacks+takeaways separately, through the tier tables | D/ST | +0.001 (−0.002, +0.005) | rejected |
| PFR pressure rates (offense allowed, defense generated) and bad-throw rate, 2018+ | D/ST | +0.001 (−0.003, +0.004) | rejected |
| Final injury report: OL / skill / defense Out+Doubtful counts | D/ST | +0.001 (−0.006, +0.007) | rejected |
| Team offense pressure count (PFR) | K | +0.005 (−0.002, +0.012) | rejected |
| Injury report counts (own offense, opposing defense) | K | −0.001 (−0.016, +0.014) | rejected |
| Sportsbook kicking-points prop (ESPN odds API), stacked on our model | K | −0.012 (−0.030, +0.006) | rejected |
| Kicking-points + FGs-made props | K | −0.007 (−0.029, +0.015) | rejected |
| Opposing QB interceptions prop | D/ST | −0.008 (−0.020, +0.005) | rejected |
| Own defenders' sacks props | D/ST | −0.002 (−0.012, +0.008) | rejected |
| Opponent team-total prop (2025+, 6 test weeks) | D/ST | +0.002 (−0.026, +0.032) | rejected |

Nothing passed, so neither model changed. The betting lines already price most of what these ideas capture (the
opposing QB's sack rate alone correlates 0.15 with D/ST points, but adds nothing once the lines are in). Since
QB-level information didn't help, paid QB sack/INT prop data isn't worth buying yet.
Pressure and injury tests (`experiments/free_data_experiments.py`, dev 2020–2024 since PFR starts in 2018)
show the same pattern: strong alone (opponent pressure rate allowed 0.17), redundant with the lines and sack-rate
form. nflverse participation data (per-play pressure) stops at 2025, so it can't feed live picks. ESPN's public odds API has free prop lines from 2024 week 2 to 2025 week 13 (ESPN BET) and 2026 (DraftKings);
`scripts/download_espn_props.py` caches them and `experiments/props_experiments.py` stacks them on our model
(expanding window, test from 2024 week 8). None helped, and on the same games the kicking-points line alone ranks
kickers worse than our model (0.178 vs 0.226). `exploration/ceiling.py` puts the theoretical correlation ceiling at ≤0.72 for both models.

2026 weeks 1–5 against other sources (`exploration/compare_sources_2026.py`): ours 0.355 vs FantasyPros top-10
experts 0.285 on D/ST (better in 5/5 weeks); 0.100 vs 0.083 on kickers. A 50/50 blend with the top-10 experts
didn't beat ours alone. Re-check once more weeks are collected.

- **ECR vs individual experts (FantasyPros API, single-expert filters, graded on the same close start/sit pairs, PPR):** in 2025 the prior year's top-10 experts averaged −0.07 pts of close-call accuracy vs ECR (4/10 ahead; only Kevin Hanson significantly, +0.54, 13/17 weeks). A consensus of just those 10 was +0.21 (CI spans 0). In 2026 Wk1–5, 2025's top-10 averaged +0.17 (8/10 ahead, none significant). Draft Sharks analysts −0.07 to −1.81 in 2025; the Draft Sharks site rankings tied ECR in 2026 (+0.02). ECR's own close-call accuracy is 53–59% week to week.
- **Rolling "top 10 so far this season" (picked each week from Weeks 1..w−1 only, all ~45 experts in the ECR pool, 20 graded weeks 2025 Wk2–17 + 2026 Wk2–5):** those 10 individually tie ECR (+0.01, 95% CI −0.34 to +0.35); the season-to-date #1 expert −0.10; a simple average-rank consensus of the 10 is +0.35 (CI −0.04 to +0.74, ahead in 14/20 weeks) — the closest thing to an edge, but not conclusive. Only ~3 of the 10 picks land in that week's actual top 10 (chance ≈ 2). Draft Sharks site vs ECR: they order 15% of close pairs differently and Draft Sharks wins exactly 50.0% of those.

## Layout

| Path | What it does |
|---|---|
| `scripts/download_data.py` | Downloads nflverse pbp, schedules + lines, and the FantasyPros ECR archive into `data/raw/` |
| `dst/dst_v2.py` | Current D/ST model: `build`, `eval` (walk-forward), `predict` (weekly board) |
| `dst/compare_fp.py` | Compares walk-forward predictions with FantasyPros weekly ECR |
| `dst/dst_v1_backtest.py` | First version: single-indicator tests, implied-total model |
| `kicker/k_model.py` | Kicker model: `build`, `eval` (walk-forward), `predict` (weekly board) |
| `kicker/k_backtest.py` | Kicker indicator backtest |
| `players/player_model.py` | QB/RB/WR/TE: `build` (ECR + pre-game usage/xFP/Vegas features), `eval` (walk-forward vs ECR) |
| `kicker/kicker-analysis-local.patch` | Local edits used to run [tdub29/Kicker-Analysis](https://github.com/tdub29/Kicker-Analysis) (missed FG = −1, output path) |
| `audit/` | Independent checks of the v1 backtests (spread sign, leakage, scoring parity) |
| `experiments/` | Feature experiments under a fixed protocol: `harness.py`, `k_experiments.py`, `dst_experiments.py` |
| `exploration/` | One-off scripts, including `compare_sources_2026.py` (ours vs ESPN, FP consensus, FP top-10 experts) |
| `scripts/weekly_snapshot.sh` | Refresh, rebuild, and save the week's lines and both boards to `snapshots/` |
| `scripts/fp_top10_grab.js` | Run on a FantasyPros weekly K/DST rankings page to capture the top-10-expert ranks |
| `inputs/` | Weekly hand inputs, e.g. wind forecasts for outdoor games |

## Weekly run

Friday and Sunday morning: `sh scripts/weekly_snapshot.sh 2026 7` (needs `inputs/2026_week7_wind.csv`). Then run
`scripts/fp_top10_grab.js` on the K and DST rankings pages and save each output line to
`inputs/fp_top10_2026.txt`, so the top-10-expert blend can be re-tested mid-season.

Or step by step:

```sh
uv run python scripts/download_data.py 2026 2026            # refresh this season + lines
uv run python -I dst/dst_v2.py build data/raw                # rebuild features
uv run python -I dst/dst_v2.py predict data/raw 2026 7 inputs/2026_week7_wind.csv
uv run python -I kicker/k_model.py build data/raw
uv run python -I kicker/k_model.py predict data/raw 2026 7 inputs/2026_week7_wind.csv
```

`inputs/<season>_week<N>_wind.csv` has columns `team,wind`: the home team of each outdoor game and its forecast wind in mph.

Kicker projections are per team. The kicker board pulls each team's current kicker from ESPN and marks `CHANGED` when that differs from whoever kicked the team's last game (for example, a signing during the week). The projection then belongs to the new kicker.

Model check (optional):

```sh
uv run python -I dst/dst_v2.py eval data/raw
uv run python -I kicker/k_model.py eval data/raw
uv run python -I dst/compare_fp.py data/raw/db_fpecr.parquet data/raw/games.csv data/dst_v2_walkforward.parquet
```

## Data sources

- [nflverse-data](https://github.com/nflverse/nflverse-data) play-by-play and [nfldata](https://github.com/nflverse/nfldata) schedules
- [DynastyProcess data](https://github.com/dynastyprocess/data) FantasyPros ECR archive and player ID crosswalk
- nflverse weekly player stats and snap counts; [ffopportunity](https://github.com/ffverse/ffopportunity) expected fantasy points
