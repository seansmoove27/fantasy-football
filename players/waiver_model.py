"""Waiver breakout model: which waiver-wire RB/WR/TE will score like a starter over the next 4 weeks.

Decision point: after week w's games (Tuesday waivers). Pool: players outside the FantasyPros ROS PPR top POOL_RANK (100)
(snapshot from the Friday before week w), a proxy for unrostered in a 14-team league.
Label: next 4 weeks' PPR points / team games >= 12 (RB/WR) or 10 (TE); missed games count as 0.
Walk-forward: each test season is fit on earlier seasons only.
Usage:
  python -I waiver_model.py eval    <data_dir>
  python -I waiver_model.py predict <data_dir> <season> <week> [rostered.csv: espn_id]  # board after <week>; drops rostered players
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(sys.argv[2]) if __name__ == '__main__' else None   # importers set waiver_model.DATA
YEARS = range(2018, 2027)
POS = ['RB', 'WR', 'TE']
POOL_RANK = 100   # FP ROS rank cutoff: ESPN free agents in this 14-team league start around FP #120
HORIZON = 4
THRESH = {'RB': 12.0, 'WR': 12.0, 'TE': 10.0}
RELOCATE = {'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'}
# snap_counts position labels: CIN backs are 'HB' (2020, 2025-26); 2020 has slash labels
POS_FIX = {'HB': 'RB', 'FB': 'RB', 'RB/W': 'RB', 'RB/F': 'RB', 'FB/R': 'RB', 'WR/R': 'WR'}
ROS_PAGE = '/nfl/rankings/ros-ppr-overall.php'
WEEKLY_PAGES = [f'/nfl/rankings/ppr-{p}.php' for p in ('rb', 'wr', 'te')]   # this week's position rank
C = 0.05   # logistic regularization (0.05 beat 0.3 / 1.0 out of sample)

FEATURES = {
    'FP ROS rank (expert)': ['log_ecr'],
    'last 2 games pts': ['ppg_last2'],
    'usage': ['snap_last2', 'snap_trend', 'tgt_last2', 'carry_last2', 'air_last2', 'ep_last2', 'vacated'],
    'all': ['log_ecr', 'ppg_last2', 'ppg_season', 'ep_last2', 'ep_season', 'snap_last2', 'snap_season', 'snap_trend',
            'tgt_last2', 'carry_last2', 'air_last2', 'vacated', 'log_pick', 'undrafted', 'years_exp', 'prior_ppg',
            'implied_next', 'log_wk_rank', 'wk_missing', 'vac_opp2'],
}


def load_ids():
    ids = pd.read_csv(DATA / 'db_playerids.csv', usecols=['gsis_id', 'pfr_id', 'fantasypros_id', 'espn_id', 'name',
                                                          'draft_year', 'draft_ovr'])
    return ids.dropna(subset=['gsis_id']).drop_duplicates('gsis_id')


def player_weeks(ids):
    out = []
    for y in YEARS:
        snaps = pd.read_parquet(DATA / f'snap_counts_{y}.parquet')
        snaps = snaps[(snaps.game_type == 'REG') & (snaps.offense_snaps > 0)]
        snaps['team'] = snaps.team.replace(RELOCATE)
        played = snaps[['season', 'week', 'team']].drop_duplicates()   # team-weeks with a game
        snaps['position'] = snaps.position.replace(POS_FIX)
        pfr = ids.dropna(subset=['pfr_id']).drop_duplicates('pfr_id')
        snaps = snaps[snaps.position.isin(POS)].merge(pfr[['pfr_id', 'gsis_id']], left_on='pfr_player_id',
                                                       right_on='pfr_id', how='left')
        # crosswalk rows missing a pfr_id: fall back to a unique name match
        by_name = ids[~ids.name.duplicated(keep=False)].set_index('name').gsis_id
        miss = snaps.gsis_id.isna()
        snaps.loc[miss, 'gsis_id'] = snaps.loc[miss, 'player'].map(by_name)
        st = pd.read_parquet(DATA / f'stats_player_week_{y}.parquet')
        st = st[st.season_type == 'REG']
        st['team'] = st.team.replace(RELOCATE)
        st = st.merge(st.groupby(['team', 'week']).carries.sum().rename('team_carries').reset_index(), on=['team', 'week'])
        st['carry_share'] = (st.carries / st.team_carries.replace(0, np.nan)).fillna(0)
        ep = pd.read_parquet(DATA / f'ep_weekly_{y}.parquet', columns=['week', 'player_id', 'total_fantasy_points_exp'])
        g = snaps.merge(st[['player_id', 'week', 'fantasy_points_ppr', 'target_share', 'air_yards_share', 'carry_share']],
                        left_on=['gsis_id', 'week'], right_on=['player_id', 'week'], how='left').drop(columns='player_id')
        g = g.merge(ep.groupby(['player_id', 'week']).total_fantasy_points_exp.sum().reset_index(),
                    left_on=['gsis_id', 'week'], right_on=['player_id', 'week'], how='left').drop(columns='player_id')
        g = g.fillna({'fantasy_points_ppr': 0, 'target_share': 0, 'air_yards_share': 0, 'carry_share': 0,
                      'total_fantasy_points_exp': 0})
        out.append((g[['season', 'week', 'gsis_id', 'pfr_player_id', 'player', 'position', 'team', 'offense_pct',
                       'fantasy_points_ppr', 'target_share', 'air_yards_share', 'carry_share', 'total_fantasy_points_exp']],
                    played))
    return pd.concat([o[0] for o in out], ignore_index=True), pd.concat([o[1] for o in out], ignore_index=True)


def implied_totals():
    g = pd.read_csv(DATA / 'games.csv')
    g = g[(g.game_type == 'REG') & g.season.isin(YEARS)]
    home = pd.DataFrame({'season': g.season, 'week': g.week, 'team': g.home_team,
                         'implied': (g.total_line + g.spread_line) / 2, 'gameday': g.gameday})
    away = pd.DataFrame({'season': g.season, 'week': g.week, 'team': g.away_team,
                         'implied': (g.total_line - g.spread_line) / 2, 'gameday': g.gameday})
    t = pd.concat([home, away])
    t['team'] = t.team.replace(RELOCATE)
    return t


def ecr_snapshots(games, fresh=False, page=ROS_PAGE):
    """FantasyPros rank on <page> (default ROS PPR overall) at decision week w.

    fresh=False: last scrape before week w+1's first game (the Friday before week w) -- what a Tuesday claim sees.
    fresh=True:  first scrape on/after week w+1's first game (Friday before week w+1) -- knows week w's results plus
                 Wed-Fri news; an upper bound for the expert baseline.
    """
    f = pd.read_parquet(DATA / 'db_fpecr.parquet', columns=['fp_page', 'id', 'ecr', 'scrape_date'])
    f = f[f.fp_page == page].copy()
    f['scrape_date'] = pd.to_datetime(f.scrape_date)
    f['id'] = pd.to_numeric(f.id, errors='coerce')
    f['rank'] = f.groupby('scrape_date').ecr.rank(method='first')
    starts = games.groupby(['season', 'week']).gameday.min().reset_index()
    starts['next_start'] = pd.to_datetime(starts.groupby('season').gameday.shift(-1))
    dates = np.sort(f.scrape_date.unique())
    rows = []
    for s in starts.dropna(subset=['next_start']).itertuples():
        t = np.datetime64(s.next_start)
        if fresh:
            i = np.searchsorted(dates, t, side='left')
            if i >= len(dates) or dates[i] - t > np.timedelta64(4, 'D'):
                continue
        else:
            i = np.searchsorted(dates, t, side='left') - 1
            if i < 0 or t - dates[i] > np.timedelta64(14, 'D'):
                continue
        rows.append((s.season, s.week, dates[i]))
    snap = pd.DataFrame(rows, columns=['season', 'week', 'scrape_date'])
    return snap.merge(f[['scrape_date', 'id', 'rank']], on='scrape_date').rename(columns={'id': 'fantasypros_id'})


def add_ranks(df, imp, fresh=False):
    """Attach the FP ROS overall rank and this week's position rank; keep only weeks with a ROS snapshot."""
    key = ['season', 'week', 'fantasypros_id']
    ecr = ecr_snapshots(imp, fresh)
    have = ecr.groupby('season').week.unique()
    df = df[[w in have.get(s, []) for s, w in zip(df.season, df.week)]]
    df = df.drop(columns=[c for c in ('rank', 'wk_rank') if c in df]).merge(ecr[key + ['rank']], on=key, how='left')
    wk = pd.concat([ecr_snapshots(imp, fresh, p) for p in WEEKLY_PAGES]).drop_duplicates(key)
    df = df.merge(wk[key + ['rank']].rename(columns={'rank': 'wk_rank'}), on=key, how='left')
    df['rank'] = df['rank'].fillna(500)
    df['log_ecr'] = np.log(df['rank'])
    df['wk_missing'] = df.wk_rank.isna().astype(int)
    worst = df.groupby(['season', 'week', 'position']).wk_rank.transform('max') + 20   # unranked: below the worst ranked
    df['log_wk_rank'] = np.log(df.wk_rank.fillna(worst).fillna(150))
    return df


def build(ids, pw, played, imp, w_range, seasons):
    """Feature rows at decision week w for every player with a snap in weeks <= w."""
    pw = pw.sort_values(['season', 'gsis_id', 'week']).copy()
    # recency-weighted (halflife 2 games) form; each row only sees games up to itself
    for src, dst in [('fantasy_points_ppr', 'ppg_ewm'), ('total_fantasy_points_exp', 'ep_ewm'), ('offense_pct', 'snap_ewm'),
                     ('target_share', 'tgt_ewm'), ('carry_share', 'carry_ewm')]:
        pw[dst] = pw.groupby(['season', 'gsis_id'])[src].transform(lambda x: x.ewm(halflife=2).mean())
    pw['opp'] = pw.target_share + pw.carry_share
    season_ppg = pw.groupby(['season', 'gsis_id']).fantasy_points_ppr.mean().rename('prior_ppg').reset_index()
    season_ppg['season'] += 1
    first = pw.groupby('pfr_player_id').season.min().rename('first_season')
    rows = []
    for s in seasons:
        ps = pw[pw.season == s]
        for w in w_range:
            past = ps[ps.week <= w]
            if past.empty or past.week.max() < w - 1:
                continue
            last2 = past.groupby('gsis_id').tail(2)
            a = past.groupby('gsis_id').agg(player=('player', 'last'), position=('position', 'last'), team=('team', 'last'),
                                            pfr_player_id=('pfr_player_id', 'last'), games=('week', 'size'),
                                            last_week=('week', 'max'), ppg_season=('fantasy_points_ppr', 'mean'),
                                            ep_season=('total_fantasy_points_exp', 'mean'),
                                            snap_season=('offense_pct', 'mean'), ppg_ewm=('ppg_ewm', 'last'),
                                            ep_ewm=('ep_ewm', 'last'), snap_ewm=('snap_ewm', 'last'),
                                            tgt_ewm=('tgt_ewm', 'last'), carry_ewm=('carry_ewm', 'last'))
            b = last2.groupby('gsis_id').agg(ppg_last2=('fantasy_points_ppr', 'mean'), ep_last2=('total_fantasy_points_exp', 'mean'),
                                             snap_last2=('offense_pct', 'mean'), tgt_last2=('target_share', 'mean'),
                                             carry_last2=('carry_share', 'mean'), air_last2=('air_yards_share', 'mean'))
            d = a.join(b).reset_index()
            d = d[d.last_week >= w - 2]                      # still active: a snap in the last 3 weeks
            d['snap_trend'] = d.snap_last2 - d.snap_season
            # vacated: same-position teammates who averaged >= 50% snaps before week w but had none in week w
            prev = ps[ps.week < w].groupby(['gsis_id', 'team', 'position']).offense_pct.mean().reset_index()
            team_played = set(played[(played.season == s) & (played.week == w)].team)
            now = set(ps[ps.week == w].gsis_id)
            gone = prev[(prev.offense_pct >= 0.5) & ~prev.gsis_id.isin(now) & prev.team.isin(team_played)]
            vac = gone.groupby(['team', 'position']).offense_pct.sum().rename('vacated').reset_index()
            d = d.merge(vac, on=['team', 'position'], how='left').fillna({'vacated': 0})
            # vacated opportunity, 2+ weeks: teammates with >= 10% target+carry share who missed weeks w-1 and w
            prev2 = ps[ps.week < w - 1].groupby(['gsis_id', 'team', 'position']).opp.mean().reset_index()
            gone2 = prev2[(prev2.opp >= 0.10) & ~prev2.gsis_id.isin(set(ps[ps.week >= w - 1].gsis_id))]
            d = d.merge(gone2.groupby(['team', 'position']).opp.sum().rename('vac_opp2').reset_index(),
                        on=['team', 'position'], how='left').fillna({'vac_opp2': 0})
            # only next week's line is posted by Tuesday's waivers (later weeks' lines would be look-ahead)
            nxt = imp[(imp.season == s) & (imp.week == w + 1)].groupby('team').implied.mean()
            d['implied_next'] = d.team.map(nxt)
            d['season'], d['week'] = s, w
            rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    df = df.merge(ids[['gsis_id', 'fantasypros_id', 'espn_id', 'draft_year', 'draft_ovr']], on='gsis_id', how='left')
    df = df.merge(season_ppg, on=['season', 'gsis_id'], how='left').merge(first, on='pfr_player_id', how='left')
    df['prior_ppg'] = df.prior_ppg.fillna(0)
    df['undrafted'] = df.draft_ovr.isna().astype(int)
    df['log_pick'] = np.log(df.draft_ovr.fillna(300))
    start = df.draft_year.where(df.draft_year.notna(), df.first_season)
    df['years_exp'] = (df.season - start).clip(lower=0).fillna(0)
    df['implied_next'] = df.implied_next.fillna(df.implied_next.mean())
    # depth-chart competition: same-position teammates' recent snap and target shares
    grp = df.groupby(['season', 'week', 'team', 'position'])
    df['comp_snap'] = grp.snap_last2.transform('sum') - df.snap_last2
    df['comp_tgt'] = grp.tgt_last2.transform('sum') - df.tgt_last2
    return df


def add_labels(df, pw, played):
    out = []
    for (s, w), d in df.groupby(['season', 'week']):
        fut = pw[(pw.season == s) & (pw.week > w) & (pw.week <= w + HORIZON)]
        pts = fut.groupby('gsis_id').fantasy_points_ppr.sum()
        tg = played[(played.season == s) & (played.week > w) & (played.week <= w + HORIZON)].groupby('team').size()
        d = d.copy()
        d['team_games'] = d.team.map(tg)
        d['next_ppg'] = d.gsis_id.map(pts).fillna(0) / d.team_games
        out.append(d)
    df = pd.concat(out)
    df = df[df.team_games >= 3]                          # skip windows cut short by season end
    df['breakout'] = (df.next_ppg >= df.position.map(THRESH)).astype(int)
    return df


def design(df, cols):
    return pd.concat([df[cols].astype(float), pd.get_dummies(df.position, dtype=float).reindex(columns=POS, fill_value=0)], axis=1)


def evaluate():
    ids = load_ids()
    pw, played = player_weeks(ids)
    imp = implied_totals()
    df = add_ranks(build(ids, pw, played, imp, range(2, 14), range(2020, 2026)), imp)
    df = add_labels(df, pw, played)
    pool = df[df['rank'] > POOL_RANK].copy()
    print(f'pool rows {len(pool)} ({pool.groupby(["season", "week"]).ngroups} season-weeks), '
          f'breakout rate {pool.breakout.mean():.3f} ({pool.breakout.sum()} breakouts)')
    res = []
    test_seasons = range(2021, 2026)
    preds = {}
    for name, cols in FEATURES.items():
        for kind in (['logistic', 'boosted'] if name == 'all' else ['logistic']):
            p = pd.Series(index=pool.index, dtype=float)
            for s in test_seasons:
                tr, te = pool[pool.season < s], pool[pool.season == s]
                if kind == 'logistic':
                    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
                else:
                    m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=40)
                m.fit(design(tr, cols), tr.breakout)
                p[te.index] = m.predict_proba(design(te, cols))[:, 1]
            preds[f'{name} ({kind})'] = p
    t = pool[pool.season.isin(test_seasons)]
    for name, p in preds.items():
        p = p[t.index]
        top = t.assign(p=p).sort_values('p', ascending=False).groupby(['season', 'week']).head(5)
        top10 = t.assign(p=p).sort_values('p', ascending=False).groupby(['season', 'week']).head(10)
        res.append({'model': name, 'AUC': roc_auc_score(t.breakout, p), 'top-5 hit rate': top.breakout.mean(),
                    'top-5 next ppg': top.next_ppg.mean(), 'top-10 hit rate': top10.breakout.mean()})
    print(f'walk-forward test {test_seasons.start}-{test_seasons.stop - 1}, {t.groupby(["season", "week"]).ngroups} weeks, '
          f'base breakout rate {t.breakout.mean():.3f}')
    print(pd.DataFrame(res).round(3).to_string(index=False))
    # per-season stability: all (logistic) vs FP rank
    a, b = preds['all (logistic)'][t.index], preds['FP ROS rank (expert) (logistic)'][t.index]

    def weekly(p, k):
        top = t.assign(p=p).sort_values('p', ascending=False).groupby(['season', 'week']).head(k)
        return top.groupby(['season', 'week']).agg(hit=('breakout', 'mean'), ppg=('next_ppg', 'mean'))
    # fairer expert baseline: the fresh FP snapshot (already knows week w), fit the same way
    # same pool and labels, ranks from the fresh snapshot: the FP-only baseline and the model refit on fresh ranks
    # (what you'd run Friday for free-agent adds after waivers clear)
    keep = ['season', 'week', 'fantasypros_id', 'rank', 'wk_rank', 'log_ecr', 'wk_missing', 'log_wk_rank']
    fr = add_ranks(pool.drop(columns=keep[3:]).reset_index(), imp, fresh=True).set_index('index')
    fr = fr.reindex(pool.index)
    fr = fr[fr.season.notna()]
    tf = t.loc[t.index.intersection(fr.index)]
    c, af = pd.Series(index=fr.index, dtype=float), pd.Series(index=fr.index, dtype=float)
    for s in test_seasons:
        tr, te = fr[fr.season < s], fr[fr.season == s]
        for out, cols in ((c, ['log_ecr']), (af, FEATURES['all'])):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(design(tr, cols), tr.breakout)
            out[te.index] = m.predict_proba(design(te, cols))[:, 1]
    print(f'\nfresh snapshot ({len(tf)} test rows): FP rank AUC {roc_auc_score(tf.breakout, c[tf.index]):.3f}, '
          f'model on fresh ranks AUC {roc_auc_score(tf.breakout, af[tf.index]):.3f}')
    rng = np.random.default_rng(0)
    t = tf
    a, b, c, af = a[t.index], b[t.index], c[t.index], af[t.index]
    for base_name, a_, base in (('stale FP rank', a, b), ('fresh FP rank', a, c), ('fresh FP rank, model on fresh ranks', af, c)):
        for k in (5, 10):
            wa, wb = weekly(a_, k), weekly(base, k)
            diff = wa - wb
            # season-block bootstrap: weeks within a season are correlated and the 4-week labels overlap
            by = {s: diff.loc[s] for s in test_seasons}
            boot = np.array([pd.concat([by[s] for s in rng.choice(list(by), len(by))]).mean().values for _ in range(5000)])
            lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
            print(f'top-{k} model minus {base_name}: hit rate {diff.hit.mean():+.3f} [{lo[0]:+.3f}, {hi[0]:+.3f}], '
                  f'next-4-wk ppg {diff.ppg.mean():+.2f} [{lo[1]:+.2f}, {hi[1]:+.2f}] (season-block 95% CI); '
                  f'ppg diff by season {[round(by[s].ppg.mean(), 2) for s in test_seasons]}')
    fit = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(design(pool, FEATURES['all']), pool.breakout)
    print('\nlogistic coefficients (standardized):')
    print(pd.Series(fit[-1].coef_[0], index=design(pool, FEATURES['all']).columns).round(2).sort_values().to_string())
    pool.to_parquet(DATA.parent / 'waiver_pool_history.parquet')


def current(ids, pw, played, imp, ecr, season, week):
    """Feature rows after <week> with the FP snapshot the training rule picks for that week, plus FP-ranked
    RB/WR/TE the model can't score (no snaps in the last 3 weeks: injured, IR, not yet debuted)."""
    snap = ecr[(ecr.season == season) & (ecr.week == week)]
    if snap.empty:
        raise SystemExit(f'no FantasyPros ROS snapshot for {season} week {week}')
    print('ROS snapshot', pd.Timestamp(snap.scrape_date.iloc[0]).date())
    cur = add_ranks(build(ids, pw, played, imp, [week], [season]), imp)
    pos = pd.read_csv(DATA / 'db_playerids.csv', usecols=['fantasypros_id', 'position']).dropna().drop_duplicates('fantasypros_id')
    ranked = snap.merge(pos, on='fantasypros_id').merge(ids[['fantasypros_id', 'name', 'espn_id']], on='fantasypros_id')
    unscored = ranked[ranked.position.isin(POS) & ~ranked.fantasypros_id.isin(cur.fantasypros_id)].sort_values('rank')
    return cur, unscored


def predict(season, week, rostered_csv):
    ids = load_ids()
    pw, played = player_weeks(ids)
    imp = implied_totals()
    ecr = ecr_snapshots(imp)
    hist = add_labels(add_ranks(build(ids, pw, played, imp, range(2, 14), range(2020, season)), imp), pw, played)
    hist = hist[hist['rank'] > POOL_RANK]
    cols = FEATURES['all']
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(design(hist, cols), hist.breakout)
    cur, unscored = current(ids, pw, played, imp, ecr, season, week)
    cur['p_breakout'] = m.predict_proba(design(cur, cols))[:, 1]
    if rostered_csv:
        rostered = pd.read_csv(rostered_csv)
        cur = cur[cur.espn_id.notna() & ~cur.espn_id.isin(rostered.espn_id)]
        unscored = unscored[~unscored.espn_id.isin(rostered.espn_id)]
    else:
        cur = cur[cur['rank'] > POOL_RANK]
    if len(unscored):
        print('FP-ranked, no snaps in the last 3 weeks (not scored):',
              ', '.join(f'{r.name} ({int(r.rank)})' for r in unscored.itertuples()), '\n')
    show = ['player', 'position', 'team', 'rank', 'games', 'ppg_season', 'ppg_last2', 'ep_last2', 'snap_last2', 'snap_trend',
            'tgt_last2', 'carry_last2', 'vacated', 'implied_next', 'p_breakout']
    cur = cur.sort_values('p_breakout', ascending=False)
    print(cur[show].head(25).round(2).to_string(index=False))
    cur[show + ['espn_id']].to_csv(DATA.parent / f'waiver_board_{season}_wk{week}.csv', index=False)


if __name__ == '__main__' and sys.argv[1] == 'eval':
    evaluate()
elif __name__ == '__main__' and sys.argv[1] == 'predict':
    predict(int(sys.argv[3]), int(sys.argv[4]), sys.argv[5] if len(sys.argv) > 5 else None)
