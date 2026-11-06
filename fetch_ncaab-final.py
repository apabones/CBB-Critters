#!/usr/bin/env python3
"""
NCAA basketball predator/prey screener: daily data job (Stage 1, early mode).

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


REQUIRED = ["game_id", "game_date", "team_id", "opponent_team_id", "team_display_name", "team_location", "team_color",
            "team_score", "opponent_team_score", "field_goals_made", "field_goals_attempted", "three_point_field_goals_made",
            "three_point_field_goals_attempted", "free_throws_attempted", "offensive_rebounds", "defensive_rebounds",
            "total_turnovers", "steals", "blocks", "assists", "points_in_paint", "turnover_points"]


def team_games(t):
    missing = [c for c in REQUIRED if c not in t.columns]
    if missing:   # fail loudly instead of quietly treating a renamed column as zeros
        raise ValueError("box score columns missing or renamed upstream: " + ", ".join(missing))
    n = lambda c: pd.to_numeric(t[c], errors="coerce").fillna(0)
    B = pd.DataFrame({"g": t.game_id.astype(str), "team": t.team_id.astype(str), "opp": t.opponent_team_id.astype(str),
                      "date": t.game_date.astype(str), "name": t.team_display_name, "short": t.team_location,
                      "color": t.team_color.astype(str), "pts": n("team_score"), "opts": n("opponent_team_score")})
    for c in ["field_goals_made", "field_goals_attempted", "three_point_field_goals_made", "three_point_field_goals_attempted",
              "free_throws_attempted", "offensive_rebounds", "defensive_rebounds", "total_turnovers", "steals", "blocks",
              "assists", "points_in_paint", "turnover_points"]:
        B[c] = n(c)
    O = B.drop(columns=["opp", "date", "name", "short", "color", "pts", "opts"]).add_prefix("o_").rename(columns={"o_g": "g"})
    M = B.merge(O, left_on=["g", "opp"], right_on=["g", "o_team"], how="left")
    have_opp = M.o_team.notna()       # opponent box missing = non-D1 opponent; those games are skipped
    M = M[have_opp].copy()
    P = lambda p: M[p + "field_goals_attempted"] - M[p + "offensive_rebounds"] + M[p + "total_turnovers"] + 0.475 * M[p + "free_throws_attempted"]
    M["poss"] = (P("") + P("o_")) / 2
    M = M[(M.poss >= 40) & (M.field_goals_attempted >= 20)]
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
    meta = B.drop_duplicates("team").set_index("team")[["name", "short", "color"]]
    return G, meta


PILLARS = {
    "Ball Security": {"TO rate": -1, "A/TO": 1, "Damage per TO": -1},
    "Pressure": {"Forced TO rate": 1, "Steal rate": 1, "Conversion per TO": 1},
    "Glass": {"ORB%": 1, "DRB%": 1},
    "Inside Attack": {"Paint pts share": 1, "FT rate": 1},
}
EXTRA = ["eFG", "ORBTV per 100", "A/TO"]


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
    pool = T[T.games >= PROVISIONAL]
    sd = pool[comps].std().replace(0, float("nan"))
    Z = (pool[comps] - pool[comps].mean()) / sd
    for p, cs in PILLARS.items():
        s = sum(Z[c] * w for c, w in cs.items()) / len(cs)
        ssd = s.std()
        T[p] = ((s - s.mean()) / ssd).round(2) if ssd and ssd > 0 else float("nan")
    T["eFG z"] = Z["eFG"].round(2)
    T = T.join(meta)
    T["stage"] = T.games.apply(lambda n: "scored" if n >= SCORED else "early" if n >= PROVISIONAL else "too early")
    return T


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


def load_slate():
    data = get(f"{ESPN}/scoreboard?dates={TODAY.strftime('%Y%m%d')}&groups=50&limit=500")
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
        games.append(dict(id=ev["id"], start=ev.get("date"), state=comp["status"]["type"].get("state"),
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
    by_name = {norm(v): k for k, v in meta["name"].items()}
    by_name.update({norm(v): k for k, v in meta["short"].items() if norm(v) not in by_name})
    flags, unmatched = {}, []
    for i in injuries:
        st = i["status"].lower()
        if not (st.startswith("out") or "doubt" in st):
            continue
        if re.search(r"season|indefinite|redshirt", (i["note"] + " " + st).lower()):
            continue
        d = parse_date(i["date"])
        fresh = d is not None and (TODAY - d).days <= 4
        tid = by_name.get(norm(i["team"]))
        if tid is None:
            unmatched.append(i["team"]); continue
        roster = pl[pl.team == tid]
        row = roster[roster.name.map(norm) == norm(i["player"])]
        if row.empty:   # fallback within this team only: same last name and first initial
            row = roster[roster.name.map(name_key) == name_key(i["player"])]
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
    T, P, meta = None, None, None
    try:
        t, P = load_boxes()
        G, meta = team_games(t)
        T = team_table(G, meta)
        status["box scores"] = f"ok: {len(G)} team-games through {G.date.max()}"
    except Exception as e:
        status["box scores"] = f"failed: {e}"
    try:
        slate = load_slate()
        status["slate"] = f"ok: {len(slate)} games, {sum(g['home_spread'] is not None for g in slate)} with spreads"
    except Exception as e:
        slate = []; status["slate"] = f"failed: {e}"
    inj = {}
    try:
        injuries = load_injuries()
        if T is not None:
            inj = wounded(injuries, P, meta)
        status["injuries"] = f"ok: {len(injuries)} listed, {sum(len(v) for v in inj.values())} matched to teams"
    except Exception as e:
        status["injuries"] = f"failed: {e}"

    teams = {}
    if T is not None:
        for tid, r in T.iterrows():
            teams[tid] = dict(name=r["name"], short=r["short"], color=hexcolor(r["color"]),
                              games=int(r.games), record=f"{int(r.wins)}-{int(r.games - r.wins)}", margin=round(float(r.margin), 1),
                              stage=r.stage, big_win=bool(r.big_win), ato=round(float(r.ato_raw), 2),
                              efg_z=None if pd.isna(r["eFG z"]) else float(r["eFG z"]),
                              orbtv=round(float(r["ORBTV per 100"]), 1),
                              pillars={p: (None if pd.isna(r[p]) else float(r[p])) for p in PILLARS})
    out_games = []
    for g in slate:
        hs = g["home_spread"]
        rec = dict(g)
        if hs is not None and hs != 0:
            fav = g["home"] if hs < 0 else g["away"]
            line = abs(hs)
            rec.update(fav=fav, dog=g["away"] if fav == g["home"] else g["home"], line=line,
                       spread_win=round(phi(line / MARGIN_SD), 3))
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
    json.dump(dict(date=str(TODAY), updated=stamp, games=out_games), open(os.path.join(OUT, "slate.json"), "w"))
    json.dump(dict(updated=stamp, teams=teams), open(os.path.join(OUT, "teams.json"), "w"), separators=(",", ":"))
    snap = {k: dict(games=v["games"], record=v["record"], stage=v["stage"], pillars=v["pillars"], efg_z=v["efg_z"], margin=v["margin"]) for k, v in teams.items()}
    json.dump(snap, open(os.path.join(OUT, "history", f"{TODAY}.json"), "w"), separators=(",", ":"))
    json.dump(dict(updated=stamp, date=str(TODAY), status=status), open(os.path.join(OUT, "latest.json"), "w"))
    for k, v in status.items():
        print(f"{k:12s} {v}")
    if T is None and not slate:
        sys.exit(1)


if __name__ == "__main__":
    main()
