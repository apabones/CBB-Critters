#!/usr/bin/env python3
"""
NCAA basketball predator/prey screener: daily data job (Stage 2).

Pulls:
  1. This season's team and player box scores (sportsdataverse, updated nightly from ESPN)
  2. Today's slate, spreads and moneylines (ESPN scoreboard)
  3. The Boyd's Bets NCAA injury report
Builds data/slate.json, data/teams.json, data/history/<date>.json and data/latest.json.

Each source is independent: if one fails, the others still save, and the page shows what's missing.
Run on GitHub (daily workflow) or locally:  python fetch_ncaab.py
"""
import csv, datetime as dt, io, json, math, os, re, sys, time, urllib.request
from html.parser import HTMLParser

import pandas as pd

SDV = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
BOYDS = "https://www.boydsbets.com/ncaa-basketball-injuries/"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
OUT = "data"
MARGIN_SD = 11.0          # typical miss of a college spread, in points
PROVISIONAL, SCORED = 2, 5  # games needed before pillars show (faded), and before flags/critters
NOW = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
TODAY = (NOW - dt.timedelta(hours=8)).date()   # Pacific calendar day
SEASON = TODAY.year + 1 if TODAY.month >= 7 else TODAY.year   # sportsdataverse labels 2026-27 as 2027
status = {}


def get(url, tries=3, raw=False):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
                b = r.read()
                return b if raw else json.loads(b)
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


# ---------------------------------------------------------------- 1. box scores
def load_boxes():
    t = pd.read_csv(io.BytesIO(get(f"{SDV}/espn_mens_college_basketball_team_boxscores/team_box_{SEASON}.csv", raw=True)), low_memory=False)
    p = pd.read_csv(io.BytesIO(get(f"{SDV}/espn_mens_college_basketball_player_boxscores/player_box_{SEASON}.csv", raw=True)), low_memory=False)
    return t, p


D1_INFO = {}


def load_d1():
    """Division I teams = teams with a D1 conference in this season's schedule (about 365)."""
    s = pd.read_csv(io.BytesIO(get(f"{SDV}/espn_mens_college_basketball_schedules/mbb_schedule_{SEASON}.csv", raw=True)), low_memory=False)
    cols = ["id", "conference_id", "display_name", "location", "color"]
    ids = pd.concat([s[[f"home_{c}" for c in cols]].set_axis(cols, axis=1), s[[f"away_{c}" for c in cols]].set_axis(cols, axis=1)])
    ids = ids[ids.conference_id.notna()].drop_duplicates("id")
    for r in ids.itertuples():   # every D1 team, even before its first D1 game
        D1_INFO[str(int(r.id))] = dict(name=r.display_name, short=r.location, color=r.color, conf=int(r.conference_id))
    return {str(int(i)): int(c) for i, c in zip(ids.id, ids.conference_id)}


REQUIRED = ["game_id", "game_date", "team_id", "opponent_team_id", "team_display_name", "team_location", "team_color",
            "team_score", "opponent_team_score", "field_goals_made", "field_goals_attempted", "three_point_field_goals_made",
            "three_point_field_goals_attempted", "free_throws_attempted", "offensive_rebounds", "defensive_rebounds",
            "total_turnovers", "steals", "blocks", "assists", "points_in_paint", "turnover_points", "fast_break_points"]


def team_games(t, d1=None):
    missing = [c for c in REQUIRED if c not in t.columns]
    if missing:   # fail loudly instead of quietly treating a renamed column as zeros
        raise ValueError("box score columns missing or renamed upstream: " + ", ".join(missing))
    n = lambda c: pd.to_numeric(t[c], errors="coerce").fillna(0)
    B = pd.DataFrame({"g": t.game_id.astype(str), "team": t.team_id.astype(str), "opp": t.opponent_team_id.astype(str),
                      "date": t.game_date.astype(str), "name": t.team_display_name, "short": t.team_location,
                      "color": t.team_color.astype(str), "pts": n("team_score"), "opts": n("opponent_team_score")})
    for c in ["field_goals_made", "field_goals_attempted", "three_point_field_goals_made", "three_point_field_goals_attempted",
              "free_throws_attempted", "offensive_rebounds", "defensive_rebounds", "total_turnovers", "steals", "blocks",
              "assists", "points_in_paint", "turnover_points", "fast_break_points"]:
        B[c] = n(c)
    O = B.drop(columns=["opp", "date", "name", "short", "color", "pts", "opts"]).add_prefix("o_").rename(columns={"o_g": "g"})
    M = B.merge(O, left_on=["g", "opp"], right_on=["g", "o_team"], how="left")
    have_opp = M.o_team.notna()       # opponent box missing = non-D1 opponent; those games are skipped
    M = M[have_opp].copy()
    P = lambda p: M[p + "field_goals_attempted"] - M[p + "offensive_rebounds"] + M[p + "total_turnovers"] + 0.475 * M[p + "free_throws_attempted"]
    M["poss"] = (P("") + P("o_")) / 2
    M = M[(M.poss >= 40) & (M.field_goals_attempted >= 20)]
    if d1:   # drop games against non-Division I opponents, the way KenPom-style ratings do
        M = M[M.team.isin(d1) & M.opp.isin(d1)]
    c1 = lambda s: s.clip(lower=1)
    G = pd.DataFrame({"g": M.g, "team": M.team, "opp": M.opp, "date": M.date, "win": (M.pts > M.opts).astype(int), "margin": M.pts - M.opts})
    # Ball Security (offense): turnover rate, A/TO, damage (points allowed per own turnover)
    G["TO rate"] = M.total_turnovers / M.poss
    G["A/TO"] = M.assists / c1(M.total_turnovers)
    G["Damage per TO"] = M.o_turnover_points / c1(M.total_turnovers)
    # Pressure (defense): forced turnover rate, steal rate, conversion (points per forced turnover)
    G["Forced TO rate"] = M.o_total_turnovers / M.poss
    G["Steal rate"] = M.steals / M.poss
    G["Conversion per TO"] = M.turnover_points / c1(M.o_total_turnovers)
    # Glass
    G["ORB%"] = M.offensive_rebounds / c1(M.offensive_rebounds + M.o_defensive_rebounds)
    G["DRB%"] = M.defensive_rebounds / c1(M.defensive_rebounds + M.o_offensive_rebounds)
    # Inside Attack (rim attempt share joins in Stage 2 from play-by-play)
    G["Paint pts share"] = M.points_in_paint / c1(M.pts)
    G["FT rate"] = M.free_throws_attempted / c1(M.field_goals_attempted)
    # shown on the card, not scored
    G["eFG"] = (M.field_goals_made + 0.5 * M.three_point_field_goals_made) / c1(M.field_goals_attempted)
    G["ORBTV per 100"] = (M.offensive_rebounds + M.o_total_turnovers - M.o_offensive_rebounds - M.total_turnovers) / M.poss * 100
    # critter traits (percentiles later): paint and glass edges, disruption, fast breaks, net rating
    G["Paint diff"] = M.points_in_paint - M.o_points_in_paint
    G["Reb diff"] = (M.offensive_rebounds + M.defensive_rebounds) - (M.o_offensive_rebounds + M.o_defensive_rebounds)
    G["Disruption"] = (M.steals + M.blocks) / M.poss
    G["FB diff"] = M.fast_break_points - M.o_fast_break_points
    G["Net per 100"] = (M.pts - M.opts) / M.poss * 100
    meta = B.drop_duplicates("team").set_index("team")[["name", "short", "color"]]
    if d1:
        meta = meta[meta.index.isin(d1)].copy()
        meta["conf"] = meta.index.map(d1)
    return G, meta


PILLARS = {
    "Ball Security": {"TO rate": -1, "A/TO": 1, "Damage per TO": -1},
    "Pressure": {"Forced TO rate": 1, "Steal rate": 1, "Conversion per TO": 1},
    "Glass": {"ORB%": 1, "DRB%": 1},
    "Inside Attack": {"Paint pts share": 1, "FT rate": 1},
}
EXTRA = ["eFG", "ORBTV per 100", "A/TO"]


CRIT = ["Paint diff", "Reb diff", "Disruption", "FB diff", "Net per 100", "Runs 12+", "Run pressure allowed", "Rim share"]


def team_table(G, meta):
    comps = [c for p in PILLARS.values() for c in p] + ["eFG", "ORBTV per 100"]
    # one-pass opponent adjustment: what each opponent typically allows, vs. the league
    adj = G.copy()
    for c in comps:
        lg = G[c].mean()
        allowed = G.groupby("opp")[c].agg(["mean", "size"])
        allowed = allowed.loc[allowed["size"] >= 2, "mean"]
        adj[c] = G[c] - (G.opp.map(allowed) - lg).fillna(0)
    agg = adj.groupby("team")[comps].mean()
    rec = G.groupby("team").agg(games=("g", "size"), wins=("win", "sum"), margin=("margin", "mean"),
                                big_win=("margin", lambda s: int((s >= 15).any())))
    T = agg.join(rec)
    T["ato_raw"] = G.groupby("team")["A/TO"].mean()
    for c in CRIT:
        T[c] = G.groupby("team")[c].mean() if c in G else float("nan")
    if "fin_chance" in G:
        f = G.groupby("team")[["fin_chance", "fin_conv"]].sum()
        T["fin_chances"], T["fin_rate"] = f.fin_chance, f.fin_conv / f.fin_chance.replace(0, float("nan"))
    pool = T[T.games >= PROVISIONAL]
    mu, sd = pool[comps].mean(), pool[comps].std().replace(0, float("nan"))
    Z = (pool[comps] - mu) / sd
    # last 5 games, on the same scale, for the Movers dashboard
    last5 = adj.sort_values("date").groupby("team").tail(5).groupby("team")[comps].mean()
    Z5 = (last5.loc[last5.index.isin(pool.index)] - mu) / sd
    rec5 = G.sort_values("date").groupby("team").tail(5).groupby("team").win.agg(["sum", "size"])
    T["last5"] = rec5["sum"].astype(int).astype(str) + "-" + (rec5["size"] - rec5["sum"]).astype(int).astype(str)
    T["last5_wins"] = rec5["sum"]
    for p, cs in PILLARS.items():
        raw = sum(Z[c] * w for c, w in cs.items()) / len(cs)
        raw5 = sum(Z5[c] * w for c, w in cs.items()) / len(cs)
        ssd = raw.std()
        T[p] = ((raw - raw.mean()) / ssd).round(2) if ssd and ssd > 0 else float("nan")
        T[p + " move"] = ((raw5 - raw) / ssd).round(2) if ssd and ssd > 0 else float("nan")
    T["eFG z"] = Z["eFG"].round(2)
    T["eFG move"] = (Z5["eFG"] - Z["eFG"]).round(2)
    T = T.join(meta)
    T["stage"] = T.games.apply(lambda n: "scored" if n >= SCORED else "early" if n >= PROVISIONAL else "too early")
    return T


# ---------------------------------------------------------------- 1b. play-by-play: runs, rim attacks, finishers
def pbp_features(d1):
    cols = ["game_id", "game_play_number", "type_text", "team_id", "scoring_play", "score_value", "home_score", "away_score",
            "home_team_id", "away_team_id", "end_game_seconds_remaining", "shooting_play"]
    d = pd.read_parquet(io.BytesIO(get(f"{SDV}/espn_mens_college_basketball_pbp/play_by_play_{SEASON}.parquet", raw=True)), columns=cols)
    d = d.dropna(subset=["game_id"])
    d["game_id"] = d.game_id.astype("int64").astype(str)
    for c in ["team_id", "home_team_id", "away_team_id"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(-1).astype("int64").astype(str)
    d = d.sort_values(["game_id", "game_play_number"])
    rows = []
    for gid, x in d.groupby("game_id", sort=False):
        h, a = x.home_team_id.iloc[0], x.away_team_id.iloc[0]
        if d1 and (h not in d1 or a not in d1):
            continue
        st = {h: dict(r12=0, rp=0, fga=0, rim=0), a: dict(r12=0, rp=0, fga=0, rim=0)}
        sc = x[x.scoring_play.fillna(False).astype(bool) & x.team_id.isin([h, a])]
        cur, run = None, 0
        for t, v in zip(sc.team_id, sc.score_value.fillna(0)):
            if t == cur:
                run += v
            else:
                if cur is not None and run >= 6:
                    st[cur]["rp"] += run - 5; st[cur]["r12"] += run >= 12
                cur, run = t, v
        if cur is not None and run >= 6:
            st[cur]["rp"] += run - 5; st[cur]["r12"] += run >= 12
        sh = x[x.shooting_play.fillna(False).astype(bool) & ~x.type_text.fillna("").str.contains("FreeThrow") & x.team_id.isin([h, a])]
        for t, ty in zip(sh.team_id, sh.type_text):
            st[t]["fga"] += 1; st[t]["rim"] += ty in ("LayUpShot", "DunkShot", "TipShot")
        tt = x.end_game_seconds_remaining.astype(float).ffill().values
        mg = (x.home_score - x.away_score).astype(float).ffill().fillna(0).values
        fin = mg[-1] if len(mg) else 0
        for t, o, sg in ((h, a, 1), (a, h, -1)):
            chance = bool(((tt <= 1200) & (sg * mg >= 15)).any())
            rows.append({"g": gid, "team": t, "Runs 12+": st[t]["r12"], "Run pressure allowed": st[o]["rp"],
                         "Rim share": st[t]["rim"] / max(1, st[t]["fga"]), "fin_chance": int(chance), "fin_conv": int(chance and sg * fin > 20)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 1c. critters
TIERS = {**{c: "Power" for c in (2, 7, 8, 23, 4)}, **{c: "Upper mid" for c in (62, 29, 3, 44, 18)},
         **{c: "Mid" for c in (11, 27, 14, 10, 9, 30, 12, 45, 24, 49, 5)}}
APEX = [0.697, 0.678, 0.64, 0.629, 0.624, 0.669]   # finisher fingerprint from the 2024-25 and 2025-26 backtests


def critters(T):
    """Species from percentiles within each conference tier, so a team is judged against its own level."""
    S = T[T.stage == "scored"].copy()
    if S.empty:
        return {}
    S["tier"] = S.conf.map(lambda c: TIERS.get(c, "Low") if not pd.isna(c) else "Low")
    traits = {"fang": "Runs 12+", "flat": "Run pressure allowed", "paint": "Paint diff", "rim": "Rim share", "reb": "Reb diff",
              "hkm": "Disruption", "fb": "FB diff", "orbtv": "ORBTV per 100", "dreb": "DRB%", "orbp": "ORB%",
              "ato": "ato_raw", "efg": "eFG", "press": "Forced TO rate", "stl": "Steal rate", "net": "Net per 100"}
    out = {}
    for tier, grp in S.groupby("tier"):
        base = grp if len(grp) >= 15 else S          # small tiers early in the season fall back to national ranks
        R = {k: base[c].rank(pct=True) if c in base and base[c].notna().any() else pd.Series(0.5, index=base.index) for k, c in traits.items()}
        prey_all = 1 - (R["reb"] + R["orbtv"] + R["fb"] + R["dreb"] + R["orbp"]) / 5
        cut = prey_all.quantile(0.8)
        for tid in grp.index:
            v = {k: round(float(R[k].get(tid, 0.5)) if not pd.isna(R[k].get(tid, 0.5)) else 0.5, 3) for k in traits}
            core = sum(x >= f for x, f in zip([v["fang"], v["paint"], v["rim"], v["reb"], v["hkm"], v["dreb"]], APEX))
            prey = round(float(prey_all.get(tid, 0.5)), 3)
            r = T.loc[tid]
            eligible = r.wins * 2 >= r.games and bool(r.big_win)       # predator minimums: winning record and a 15-point win
            sp = "Blob"
            if (core >= 4 or v["net"] >= 0.85) and eligible:
                if v["fang"] >= 0.9 and v["net"] >= 0.97 and core >= 5: sp = "Saber-tooth"
                elif v["ato"] >= 0.85 and v["efg"] >= 0.85 and v["net"] >= 0.85: sp = "Gorilla"
                elif v["reb"] >= 0.8 and v["paint"] >= 0.75: sp = "Lion"
                elif v["press"] >= 0.7 and v["fb"] >= 0.6: sp = "Wolf"
                elif (v["press"] >= 0.7 or v["stl"] >= 0.75) and v["reb"] < 0.6: sp = "Weasel"
                elif core >= 5: sp = "Tiger"
                else: sp = "Jackal"
            elif prey >= cut:
                sp = "Rabbit" if v["ato"] < 0.4 else "Goat"
            elif prey >= 0.6:
                sp = "Hare" if v["ato"] < 0.4 else "Ram"
            out[tid] = dict(sp=sp, tier=tier, core=int(core), prey=prey, **{k: v[k] for k in traits if k not in ("orbp",)})
    return out


def movers(teams, today):
    """Compare each team to its snapshot from about a week ago, plus last-5 pillar moves."""
    hist = sorted(f[:-5] for f in os.listdir(os.path.join(OUT, "history")) if f.endswith(".json"))
    old = [h for h in hist if h <= str(today - dt.timedelta(days=7))]
    prev = json.load(open(os.path.join(OUT, "history", old[-1] + ".json"))) if old else {}
    out = []
    for tid, t in teams.items():
        if t["stage"] != "scored":
            continue
        then = (prev.get(tid) or {}).get("species")
        jumps = {p: m for p, m in (t.get("moves") or {}).items() if m is not None and abs(m) >= 0.75}
        up = any(m > 0 for m in jumps.values())
        hot = (t.get("last5_wins") or 0) >= 4
        changed = then is not None and then != t.get("species")
        if not (changed or jumps or hot):
            continue
        if hot and up: label = "Real shift"
        elif hot and (t.get("efg_move") or 0) >= 0.75: label = "Hot streak (shooting)"
        elif hot: label = "Hot streak"
        else: label = ""
        out.append(dict(team=tid, name=t["name"], then=then, now=t.get("species"), jumps=jumps, last5=t.get("last5"), label=label,
                        since=old[-1] if old else None))
    out.sort(key=lambda m: (m["then"] is None or m["then"] == m["now"], -len(m["jumps"])))
    return out


# ---------------------------------------------------------------- 2. slate + lines
def num(x):
    try:
        return float(str(x).replace("+", "").replace("PK", "0"))
    except Exception:
        return None


def ml_from(odds, side):
    o = odds.get(side + "TeamOdds") or {}
    v = num(o.get("moneyLine"))
    if v is None:
        try:
            v = num(odds["moneyline"][side]["close"]["odds"])
        except Exception:
            v = None
    return v


def implied(ml):
    if ml is None or ml == 0:
        return None
    return -ml / (-ml + 100) if ml < 0 else 100 / (ml + 100)


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def load_slate(day):
    data = get(f"{ESPN}/scoreboard?dates={day.strftime('%Y%m%d')}&groups=50&limit=500")
    games = []
    for ev in data.get("events", []):
        comp = ev["competitions"][0]
        tm = {c["homeAway"]: c for c in comp["competitors"]}
        h, a = tm.get("home"), tm.get("away")
        if not h or not a:
            continue
        odds = (comp.get("odds") or [None])[0] or {}
        hs = num(odds.get("spread"))                       # home team's spread (negative = home favored)
        mlh, mla = ml_from(odds, "home"), ml_from(odds, "away")
        games.append(dict(id=ev["id"], day=str(day), start=ev.get("date"), state=comp["status"]["type"].get("state"),
                          neutral=bool(comp.get("neutralSite")), home=str(h["team"]["id"]), away=str(a["team"]["id"]),
                          home_name=h["team"].get("displayName", ""), away_name=a["team"].get("displayName", ""),
                          details=odds.get("details", ""), home_spread=hs, ml_home=mlh, ml_away=mla))
    return games


# ---------------------------------------------------------------- 3. injuries
class Table(HTMLParser):
    def __init__(self):
        super().__init__(); self.rows, self.row, self.cell, self.inc = [], None, "", False
    def handle_starttag(self, tag, attrs):
        if tag == "tr": self.row = []
        elif tag in ("td", "th") and self.row is not None: self.inc, self.cell = True, ""
    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.row is not None: self.row.append(" ".join(self.cell.split())); self.inc = False
        elif tag == "tr" and self.row is not None:
            if self.row: self.rows.append(self.row)
            self.row = None
    def handle_data(self, d):
        if self.inc: self.cell += d


def load_injuries():
    tp = Table(); tp.feed(get(BOYDS, raw=True).decode("utf-8", "ignore"))
    out, team = [], ""
    for r in tp.rows:
        low = [c.lower() for c in r]
        if "player" in low and "status" in low:
            continue                                       # header row
        if len(r) == 1:
            team = r[0]; continue                          # team section header
        if len(r) >= 6:
            team, player, pos, st, date, note = r[:6]
        elif len(r) == 5:
            player, pos, st, date, note = r
        else:
            continue
        out.append(dict(team=team, player=player, pos=pos, status=st, date=date, note=note))
    return out


SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?", re.I)


def norm(s):
    s = SUFFIX.sub("", str(s).lower().replace("’", "").replace("'", ""))
    return re.sub(r"[^a-z0-9]", "", s)


def name_key(s):
    """last name + first initial, for matching 'A.J. Smith Jr.' to 'AJ Smith' or 'Cam Jones' to 'Cameron Jones'"""
    parts = [p for p in re.sub(r"[^a-z ]", "", SUFFIX.sub("", str(s).lower())).split() if p]
    return (parts[-1], parts[0][0]) if len(parts) >= 2 else (norm(s), "")


def parse_date(s):
    for fmt in ("%b %d", "%B %d", "%m/%d", "%m/%d/%Y", "%Y-%m-%d", "%b %d, %Y", "%B %d, %Y"):
        try:
            d = dt.datetime.strptime(s.strip().replace(".", ""), fmt).date()
            if d.year == 1900:
                d = d.replace(year=TODAY.year if d.month <= TODAY.month + 1 else TODAY.year - 1)
            return d
        except Exception:
            pass
    return None


def wounded(injuries, P, meta):
    """Out/Doubtful, reported in the last 4 days, not season-long, and a top-2 scorer this season."""
    n = lambda c: pd.to_numeric(P[c], errors="coerce").fillna(0)
    pl = pd.DataFrame({"team": P.team_id.astype(str), "name": P.athlete_display_name, "pts": n("points"), "min": n("minutes")})
    pl = pl[pl["min"] > 0]
    team_g = P[pd.to_numeric(P.minutes, errors="coerce").fillna(0) > 0].groupby(P.team_id.astype(str)).game_id.nunique()
    pl = pl.groupby(["team", "name"]).pts.agg(["mean", "size"]).reset_index()
    # a one-game hot bench guy can't outrank the regulars: must have played at least half the team's games
    pl = pl[pl["size"] >= (pl.team.map(team_g).fillna(1) / 2).clip(lower=1)].copy()
    pl["rank"] = pl.groupby("team")["mean"].rank(ascending=False, method="first")
    names = {k: (v["name"], v["short"]) for k, v in D1_INFO.items()} or {k: (r["name"], r["short"]) for k, r in meta.iterrows()}
    by_name = {norm(n): k for k, (n, sh) in names.items()}
    by_short = {norm(sh): k for k, (n, sh) in names.items() if isinstance(sh, str)}

    def find_team(label):
        key = norm(label)
        if key in by_name:
            return by_name[key]
        # 'UNLV Runnin' Rebels' vs ESPN's 'UNLV Rebels': match the longest school name the label starts with
        hits = [sh for sh in by_short if sh and key.startswith(sh)]
        return by_short[max(hits, key=len)] if hits else None
    flags, unmatched = {}, []
    for i in injuries:
        st = i["status"].lower()
        if not (st.startswith("out") or "doubt" in st):
            continue
        if re.search(r"season|indefinite|redshirt", (i["note"] + " " + st).lower()):
            continue
        d = parse_date(i["date"])
        fresh = d is not None and (TODAY - d).days <= 4
        tid = find_team(i["team"])
        if tid is None:
            unmatched.append(i["team"]); continue
        roster = pl[pl.team == tid]
        row = roster[[norm(n) == norm(i["player"]) for n in roster.name]]
        if row.empty:   # fallback within this team only: same last name and first initial
            key = name_key(i["player"])
            row = roster[[name_key(n) == key for n in roster.name]]
        rank = int(row["rank"].iloc[0]) if len(row) else None
        ppg = round(float(row["mean"].iloc[0]), 1) if len(row) else None
        flags.setdefault(tid, []).append(dict(player=i["player"], status=i["status"], date=i["date"], fresh=fresh,
                                              rank=rank, ppg=ppg, top2=bool(rank and rank <= 2)))
    if unmatched:
        status["injury teams not matched"] = ", ".join(sorted(set(unmatched)))[:300]
    return flags


def hexcolor(c):
    c = str(c).lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    return "#" + c[:6] if re.fullmatch(r"[0-9a-fA-F]{6}", c[:6] or "") else "#888888"


# ---------------------------------------------------------------- assemble
def main():
    os.makedirs(os.path.join(OUT, "history"), exist_ok=True)
    T, P, meta, d1 = None, None, None, None
    try:
        t, P = load_boxes()
        try:
            d1 = load_d1()
        except Exception as e:
            status["D1 list"] = f"failed, non-D1 games kept: {e}"
        G, meta = team_games(t, d1)
        try:
            F = pbp_features(d1)
            G = G.merge(F, on=["g", "team"], how="left")
            status["play-by-play"] = f"ok: {F.g.nunique()} games"
        except Exception as e:
            status["play-by-play"] = f"failed, critters use box scores only: {e}"
        T = team_table(G, meta)
        status["box scores"] = f"ok: {len(G)} team-games through {G.date.max()}"
    except Exception as e:
        status["box scores"] = f"failed: {e}"
    slate = []
    for k in range(3):
        day = TODAY + dt.timedelta(days=k)
        try:
            slate += load_slate(day)
        except Exception as e:
            status[f"slate {day}"] = f"failed: {e}"
    status["slate"] = f"{len(slate)} games over 3 days, {sum(g['home_spread'] is not None for g in slate)} with spreads"
    inj = {}
    try:
        injuries = load_injuries()
        if T is not None:
            inj = wounded(injuries, P, meta)
        status["injuries"] = f"ok: {len(injuries)} listed, {sum(len(v) for v in inj.values())} matched to teams"
    except Exception as e:
        status["injuries"] = f"failed: {e}"

    teams, crit = {}, {}
    if T is not None:
        try:
            crit = critters(T)
        except Exception as e:
            status["critters"] = f"failed: {e}"
        f = lambda x, d=1: None if x is None or pd.isna(x) else round(float(x), d)
        for tid, r in T.iterrows():
            c = crit.get(tid, {})
            teams[tid] = dict(name=r["name"], short=r["short"], color=hexcolor(r["color"]), conf=None if pd.isna(r.get("conf")) else int(r["conf"]),
                              games=int(r.games), record=f"{int(r.wins)}-{int(r.games - r.wins)}", margin=f(r.margin),
                              stage=r.stage, big_win=bool(r.big_win), ato=f(r.ato_raw, 2), efg_z=f(r["eFG z"], 2), orbtv=f(r["ORBTV per 100"]),
                              last5=r.get("last5"), last5_wins=None if pd.isna(r.get("last5_wins")) else int(r["last5_wins"]),
                              efg_move=f(r.get("eFG move"), 2), fin=f(r.get("fin_rate"), 2), fin_n=None if pd.isna(r.get("fin_chances", float("nan"))) else int(r["fin_chances"]),
                              pillars={p: f(r[p], 2) for p in PILLARS}, moves={p: f(r[p + " move"], 2) for p in PILLARS},
                              species=c.get("sp", "Blob"), critter=c or None)
    for tid, info in D1_INFO.items():
        if tid not in teams:
            teams[tid] = dict(name=info["name"], short=info["short"], color=hexcolor(info["color"]), conf=info["conf"], games=0, record="0-0",
                              margin=None, stage="too early", big_win=False, ato=None, efg_z=None, orbtv=None, last5=None, last5_wins=None,
                              efg_move=None, fin=None, fin_n=None, pillars={p: None for p in PILLARS}, moves={p: None for p in PILLARS},
                              species="Blob", critter=None)
    out_games = []
    for g in slate:
        hs = g["home_spread"]
        rec = dict(g)
        if hs is not None and hs != 0:
            fav = g["home"] if hs < 0 else g["away"]
            line = abs(hs)
            rec.update(fav=fav, dog=g["away"] if fav == g["home"] else g["home"], line=line, spread_win=round(phi(line / MARGIN_SD), 3))
            mlf = g["ml_home"] if fav == g["home"] else g["ml_away"]
            mld = g["ml_away"] if fav == g["home"] else g["ml_home"]
            pf, pd_ = implied(mlf), implied(mld)
            if pf and pd_:
                rec["ml_win"] = round(pf / (pf + pd_), 3)             # vig removed
        out_games.append(rec)
    for tid, v in inj.items():
        if tid in teams:
            teams[tid]["injuries"] = v

    stamp = NOW.isoformat(timespec="seconds") + "Z"
    mv = movers(teams, TODAY) if teams else []
    json.dump(dict(date=str(TODAY), updated=stamp, games=out_games), open(os.path.join(OUT, "slate.json"), "w"))
    json.dump(dict(updated=stamp, teams=teams), open(os.path.join(OUT, "teams.json"), "w"), separators=(",", ":"))
    json.dump(dict(updated=stamp, movers=mv), open(os.path.join(OUT, "movers.json"), "w"))
    snap = {k: dict(games=v["games"], record=v["record"], stage=v["stage"], species=v["species"], pillars=v["pillars"],
                    efg_z=v["efg_z"], margin=v["margin"]) for k, v in teams.items()}
    json.dump(snap, open(os.path.join(OUT, "history", f"{TODAY}.json"), "w"), separators=(",", ":"))
    json.dump(dict(updated=stamp, date=str(TODAY), status=status), open(os.path.join(OUT, "latest.json"), "w"))
    for k, v in status.items():
        print(f"{k:14s} {v}")
    if T is None and not slate:
        sys.exit(1)


if __name__ == "__main__":
    main()
