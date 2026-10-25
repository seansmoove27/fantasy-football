"""Download pre-game sportsbook prop lines from ESPN's public odds API (props exist from 2024 on).

Keeps the markets the D/ST and kicker models can use: kicking points, field goals / extra points made, team total
points, passing interceptions, defensive interceptions, sacks. One cached JSON per game in data/raw/espn_props/,
then a flat table data/raw/espn_props.parquet with one row per (game, market, player/team, line).

Usage: python -I download_espn_props.py <first_season> <last_season> [--refresh-upcoming]
Unofficial endpoint: requests are throttled and cached; already-played games are never re-fetched.
"""
import json
import socket
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / 'data' / 'raw'
CACHE = RAW / 'espn_props'
API = 'https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{0}/competitions/{0}/odds'
KEEP = ('Total Kicking Points', 'Total Field Goals Made', 'Total Extra Points Made', 'Team Total Points',
        'Total Passing Interceptions', 'To throw an interception', 'Total Defensive Interceptions', 'Total Sacks')
PAUSE = 0.35  # seconds between requests
socket.setdefaulttimeout(20)  # a dropped connection otherwise blocks forever


def get(url):
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:  # provider has no props for this game
                return {}
            err = e
            time.sleep(2 * (attempt + 1))
        except Exception as e:  # noqa: BLE001 - retry any network/HTTP error, then give up on this game
            err = e
            time.sleep(2 * (attempt + 1))
    raise err


def fetch_game(espn_id):
    base = API.format(espn_id)
    time.sleep(PAUSE)
    providers = [it['provider'] for it in get(base).get('items', []) if 'Live' not in it['provider']['name']]
    out = {'providers': [p['name'] for p in providers], 'items': []}
    for p in providers:
        page, items = 1, []
        while True:
            time.sleep(PAUSE)
            d = get(f'{base}/{p["id"]}/propBets?limit=1000&page={page}')
            items += d.get('items', [])
            if page >= d.get('pageCount', 1):
                break
            page += 1
        if items:  # first non-live provider with props
            out['provider'] = p['name']
            out['items'] = [it for it in items if it.get('type', {}).get('name', '').startswith(KEEP)]
            break
    return out


def ref_id(ref):
    return ref['$ref'].split('/')[-1].split('?')[0] if ref else None


def odds(side):
    """American odds as a float; 'Even'/'EVEN' is +100."""
    if not side or side.get('american') in (None, ''):
        return None
    a = str(side['american']).strip()
    return 100.0 if a.lower() == 'even' else float(a)


def flatten(game_id, d):
    rows = []
    for it in d['items']:
        cur, opn = it.get('current') or {}, it.get('open') or {}
        rows.append(dict(game_id=game_id, provider=d.get('provider'), market=it['type']['name'],
                         espn_athlete=ref_id(it.get('athlete')), espn_team=ref_id(it.get('team')),
                         line=(cur.get('target') or {}).get('value'), over=odds(cur.get('over')), under=odds(cur.get('under')),
                         open_line=(opn.get('target') or {}).get('value'), open_over=odds(opn.get('over')),
                         open_under=odds(opn.get('under'))))
    return rows


if __name__ == '__main__':
    first, last = int(sys.argv[1]), int(sys.argv[2])
    refresh = '--refresh-upcoming' in sys.argv
    CACHE.mkdir(parents=True, exist_ok=True)
    g = pd.read_csv(RAW / 'games.csv')
    g = g[(g.season.between(first, last)) & (g.game_type == 'REG') & g.espn.notna()]
    g = g[pd.to_datetime(g.gameday) <= pd.Timestamp(date.today() + timedelta(days=7))]
    rows, failed = [], []
    for n, r in enumerate(g.itertuples(), 1):
        path = CACHE / f'{r.game_id}.json'
        upcoming = pd.Timestamp(r.gameday) >= pd.Timestamp(date.today())
        if not path.exists() or (refresh and upcoming):
            try:
                path.write_text(json.dumps(fetch_game(int(r.espn))))
            except Exception as e:  # noqa: BLE001
                failed.append(r.game_id)
                print('failed', r.game_id, e, file=sys.stderr)
                continue
        rows += flatten(r.game_id, json.loads(path.read_text()))
        if n % 25 == 0:
            print(f'{n}/{len(g)} games', file=sys.stderr, flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(RAW / 'espn_props.parquet')
    print(f'wrote {len(df)} prop rows for {df.game_id.nunique()} games; failed {len(failed)}: {failed[:10]}')
    print(df.groupby(['market']).game_id.nunique().to_string())
