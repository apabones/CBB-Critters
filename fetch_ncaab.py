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
import csv, datetime as dt, html, io, json, math, os, re, sys, time, unicodedata, urllib.parse, urllib.request
from html.parser import HTMLParser

import numpy as np
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


METHOD = "2026-11-16"     # bump when the rating method changes, so Movers doesn't report the change as teams moving
SHRINK_K = 3              # an opponent with k other games gets half weight in the schedule adjustment
ZERO_FLAGS = []           # box score values that look wrong, for a manual check
POOL = {"min": 3, "n": 0}


def team_games(t, d1=None):
    missing = [c for c in REQUIRED if c not in t.columns]
    if missing:   # fail loudly instead of quietly treating a renamed column as zeros
        raise ValueError("box score columns missing or renamed upstream: " + ", ".join(missing))
    n = lambda c: pd.to_numeric(t[c], errors="coerce")       # a blank stays blank instead of becoming 0
    B = pd.DataFrame({"g": t.game_id.astype(str), "team": t.team_id.astype(str), "opp": t.opponent_team_id.astype(str),
                      "date": t.game_date.astype(str), "name": t.team_display_name, "short": t.team_location,
                      "color": t.team_color.astype(str), "pts": n("team_score"), "opts": n("opponent_team_score")})
    for c in ["field_goals_made", "field_goals_attempted", "three_point_field_goals_made", "three_point_field_goals_attempted",
              "free_throws_attempted", "offensive_rebounds", "defensive_rebounds", "total_turnovers", "steals", "blocks",
              "assists", "points_in_paint", "turnover_points", "fast_break_points"]:
        B[c] = n(c)
    B = B.drop_duplicates(["g", "team"])
    names = B.drop_duplicates("team").set_index("team").short.to_dict()

    def pair(B):
        O = B.drop(columns=["opp", "date", "name", "short", "color", "pts", "opts"]).add_prefix("o_").rename(columns={"o_g": "g"})
        M = B.merge(O, left_on=["g", "opp"], right_on=["g", "o_team"], how="left")
        M = M[M.o_team.notna()].copy()       # opponent box missing = non-D1 opponent; those games are skipped
        if d1:   # drop games against non-Division I opponents, the way KenPom-style ratings do
            M = M[M.team.isin(d1) & M.opp.isin(d1)]
        return M
    M = pair(B)
    # implausible zeros: blank the stat (for both teams' views of the game) and list it for a manual check
    two = M.field_goals_made - M.three_point_field_goals_made
    rules = [("points_in_paint", (M.points_in_paint == 0) & (two >= 5), lambda r: f"{int(r.field_goals_made - r.three_point_field_goals_made)} two-point makes", True),
             ("turnover_points", (M.turnover_points == 0) & (M.o_total_turnovers >= 6), lambda r: f"opponent had {int(r.o_total_turnovers)} turnovers", True),
             ("assists", (M.assists == 0) & (M.field_goals_made >= 10), lambda r: f"{int(r.field_goals_made)} made baskets", True),
             ("fast_break_points", (M.fast_break_points == 0) & (M.points_in_paint == 0) & (M.turnover_points == 0), lambda r: "paint and turnover points also 0", True),
             ("total_turnovers", M.total_turnovers <= 2, lambda r: "very low, possible scoring error", False),
             ("steals", M.steals == 0, lambda r: "no steals", False),
             ("offensive_rebounds", M.offensive_rebounds == 0, lambda r: "no offensive rebounds", False)]
    ZERO_FLAGS.clear()
    blank = []
    for col, mask, why, do_blank in rules:
        for r in M[mask.fillna(False)].itertuples():
            ZERO_FLAGS.append(dict(game=r.g, date=r.date[:10], team=names.get(r.team, r.team), opp=names.get(r.opp, r.opp), stat=col,
                                   value=None if pd.isna(getattr(r, col)) else int(getattr(r, col)), why=why(r), blanked=do_blank))
            if do_blank:
                blank.append((r.g, r.team, col))
    for c in ["field_goals_attempted", "offensive_rebounds", "total_turnovers", "free_throws_attempted", "team_score"]:
        cc = "pts" if c == "team_score" else c
        for r in M[M[cc].isna()].itertuples():
            ZERO_FLAGS.append(dict(game=r.g, date=r.date[:10], team=names.get(r.team, r.team), opp=names.get(r.opp, r.opp), stat=c,
                                   value=None, why="missing, so the game is left out", blanked=True))
    if blank:
        for g, tm, col in blank:
            B.loc[(B.g == g) & (B.team == tm), col] = float("nan")
        M = pair(B)
    P = lambda p: M[p + "field_goals_attempted"] - M[p + "offensive_rebounds"] + M[p + "total_turnovers"] + 0.475 * M[p + "free_throws_attempted"]
    M["poss"] = (P("") + P("o_")) / 2
    M = M[(M.poss >= 40) & (M.field_goals_attempted >= 20)]     # a blank core stat makes poss blank, which drops the game
    c1 = lambda s: s.clip(lower=1)
    G = pd.DataFrame({"g": M.g, "team": M.team, "opp": M.opp, "date": M.date, "win": (M.pts > M.opts).astype(int), "margin": M.pts - M.opts})
    # totals kept per game so ratio stats can be built from season sums instead of averaging tiny-denominator games
    G["_ast"], G["_to"], G["_otp"], G["_tp"], G["_oto"] = M.assists, M.total_turnovers, M.o_turnover_points, M.turnover_points, M.o_total_turnovers
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
    # Inside Attack
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
RATIO = {"A/TO": ("_ast", "_to"), "Damage per TO": ("_otp", "_to"), "Conversion per TO": ("_tp", "_oto")}
ADJ = {}    # per-game opponent-adjusted values, shared with the apex step


def opp_offsets(G, cols, k=SHRINK_K):
    """How much each game's number owes to the opponent: what that opponent allowed in its OTHER games, vs the league,
    shrunk toward zero when the opponent has few other games (weight = n / (n + k))."""
    off = pd.DataFrame(index=G.index)
    for c in cols:
        if c not in G:
            continue
        if c in RATIO:
            a, b = RATIO[c]
            ok = G[a].notna() & G[b].notna()
            nu, de = G[a].where(ok, 0), G[b].where(ok, 0)
            league = nu.sum() / max(1, de.sum())
            sn = nu.groupby(G.opp).transform("sum") - nu
            sd_ = de.groupby(G.opp).transform("sum") - de
            cnt = ok.astype(int).groupby(G.opp).transform("sum") - ok.astype(int)
            allowed = sn / sd_.where(sd_ > 0)
        else:
            v = G[c]
            ok = v.notna()
            league = v.mean()
            s = v.fillna(0).groupby(G.opp).transform("sum") - v.fillna(0)
            cnt = ok.astype(int).groupby(G.opp).transform("sum") - ok.astype(int)
            allowed = s / cnt.where(cnt > 0)
        off[c] = (cnt / (cnt + k) * (allowed - league)).fillna(0)
    return off


def team_values(G, off, cols):
    """Season (or window) values per team, schedule-adjusted. Ratio stats come from summed totals."""
    out = {}
    for c in cols:
        if c not in G:
            continue
        if c in RATIO:
            a, b = RATIO[c]
            ok = G[a].notna() & G[b].notna()
            tot = G[ok].groupby("team")[[a, b]].sum()
            out[c] = tot[a] / tot[b].clip(lower=1) - off.loc[ok, c].groupby(G.team[ok]).mean()
        else:
            ok = G[c].notna()
            out[c] = (G[c] - off[c])[ok].groupby(G.team[ok]).mean()
    return pd.DataFrame(out)


def pct_vs(base, s):
    """Percentile of each value in s against the values in base. Matches rank(pct=True) for teams inside base."""
    b = np.sort(pd.Series(base).dropna().values)
    s = pd.Series(s)
    if not len(b):
        return pd.Series(float("nan"), index=s.index)
    lo, hi = np.searchsorted(b, s.values, "left"), np.searchsorted(b, s.values, "right")
    return pd.Series(np.where(s.isna(), np.nan, (lo + hi + 1) / (2 * len(b))), index=s.index).clip(upper=1)


def team_table(G, meta):
    comps = [c for p in PILLARS.values() for c in p] + ["eFG", "ORBTV per 100"]
    cols = comps + [c for c in CRIT if c in G]
    off = opp_offsets(G, cols)
    ADJ.clear(); ADJ["off"] = off
    ADJ["G"] = G[["g", "team", "opp"]].join(pd.DataFrame({c: G[c] - off[c] for c in off}))
    T = team_values(G, off, cols)
    rec = G.groupby("team").agg(games=("g", "size"), wins=("win", "sum"), margin=("margin", "mean"),
                                big_win=("margin", lambda s: int((s >= 15).any())))
    T = rec.join(T)
    tot = G[G._ast.notna() & G._to.notna()].groupby("team")[["_ast", "_to"]].sum()
    T["ato_raw"] = tot._ast / tot._to.clip(lower=1)          # the real season A/TO, shown on the card
    for c in CRIT:
        if c not in T:
            T[c] = float("nan")
    if "fin_chance" in G:
        f = G.groupby("team")[["fin_chance", "fin_conv"]].sum()
        T["fin_chances"], T["fin_rate"] = f.fin_chance, f.fin_conv / f.fin_chance.replace(0, float("nan"))
    # the scale is set by teams with enough games (3+ early, 5+ once 200 teams get there); 2-game teams are placed on it
    POOL["min"] = SCORED if (T.games >= SCORED).sum() >= 200 else 3
    pool = T[T.games >= POOL["min"]]
    POOL["n"] = int(len(pool))
    rated = T[T.games >= PROVISIONAL]
    mu, sd = pool[comps].mean(), pool[comps].std().replace(0, float("nan"))
    Z = (rated[comps] - mu) / sd
    # last 5 games, on the same scale, for the Movers dashboard
    Gs = G.sort_values("date")
    l5 = Gs.groupby("team").tail(5).index
    last5 = team_values(G.loc[l5], off.loc[l5], comps)
    Z5 = (last5.loc[last5.index.isin(rated.index)] - mu) / sd
    rec5 = Gs.groupby("team").tail(5).groupby("team").win.agg(["sum", "size"])
    T["last5"] = rec5["sum"].astype(int).astype(str) + "-" + (rec5["size"] - rec5["sum"]).astype(int).astype(str)
    T["last5_wins"] = rec5["sum"]
    for p, cs in PILLARS.items():
        raw = sum(Z[c] * w for c, w in cs.items()) / len(cs)
        raw5 = sum(Z5[c] * w for c, w in cs.items()) / len(cs)
        rp = raw[raw.index.isin(pool.index)]
        m_, ssd = rp.mean(), rp.std()
        T[p] = ((raw - m_) / ssd).round(2) if ssd and ssd > 0 else float("nan")
        T[p + " move"] = ((raw5 - raw) / ssd).round(2) if ssd and ssd > 0 else float("nan")
    T["eFG z"] = Z["eFG"].round(2)
    T["eFG move"] = (Z5["eFG"] - Z["eFG"]).round(2)
    T = T.join(meta)
    T["tier"] = T.conf.map(lambda c: TIERS.get(c, "Low") if not pd.isna(c) else "Low")
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


TIER_MIN = 30   # a conference tier needs this many rated teams before it becomes the comparison pool
TIER_FLOOR = 10  # a locked tier falls back to the national pool only if it ever has fewer than this


def tier_pools(T, pool_state):
    """Comparison pool for each tier. A tier switches from national to its own pool the first day it has TIER_MIN
    rated teams, and then stays switched (the lock is saved), so percentiles never flip back and forth."""
    pool = T[T.games >= POOL["min"]]
    locked = pool_state.setdefault("tiers_locked", {})
    out = {}
    for tier in sorted(set(T.tier)):
        tp = pool[pool.tier == tier]
        if tier not in locked and len(tp) >= TIER_MIN:
            locked[tier] = str(TODAY)
        out[tier] = (tp, "tier") if tier in locked and len(tp) >= TIER_FLOOR else (pool, "national")
    return out


def critters(T, pool_state):
    """Species for teams with 5+ games, from schedule-adjusted traits ranked within the team's tier pool."""
    S = T[T.stage == "scored"]
    if S.empty:
        return {}
    pools = tier_pools(T, pool_state)
    traits = {"fang": "Runs 12+", "flat": "Run pressure allowed", "paint": "Paint diff", "rim": "Rim share", "reb": "Reb diff",
              "hkm": "Disruption", "fb": "FB diff", "orbtv": "ORBTV per 100", "dreb": "DRB%", "orbp": "ORB%",
              "ato": "A/TO", "efg": "eFG", "press": "Forced TO rate", "stl": "Steal rate", "net": "Net per 100"}
    out = {}
    for tier, grp in S.groupby("tier"):
        base, kind = pools[tier]
        X = T.loc[base.index.union(grp.index)]
        R = {k: pct_vs(base[c], X[c]) if c in base and base[c].notna().any() else pd.Series(0.5, index=X.index) for k, c in traits.items()}
        prey_all = 1 - (R["reb"] + R["orbtv"] + R["fb"] + R["dreb"] + R["orbp"]) / 5
        cut = prey_all.reindex(base.index).quantile(0.8)
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
            out[tid] = dict(sp=sp, tier=tier, pool=kind, pool_n=int(len(base)), core=int(core), prey=prey,
                            **{k: v[k] for k in traits if k not in ("orbp",)})
    return out


APEX_TRAITS = [("runs", "Runs 12+"), ("paint", "Paint diff"), ("rim", "Rim share"), ("glass", "Reb diff"), ("disrupt", "Disruption"), ("dreb", "DRB%")]
APEX_CUT = dict(zip([k for k, _ in APEX_TRAITS], APEX))   # same percentile lines as the species fingerprint


def apex_profile(T, G, pool_state):
    """For every team with 2+ games: each apex trait's percentile (same pools as the species), how consistently it shows
    up game to game (games vs bottom-25% opponents count half), its split vs top-half opponents, and an overall rank.
    The game-by-game line sits at the same percentile of single games, so an average team clears it about 30% of the time."""
    rated = T[T.games >= PROVISIONAL]
    if rated.empty:
        return {}
    pools = tier_pools(T, pool_state)
    GA = ADJ["G"]
    # opponent strength without the game being rated: the opponent's adjusted net from its other games
    netc = "Net per 100"
    s = GA.groupby("team")[netc].agg(["sum", "count"])
    mirror = GA[["g", "team", netc]].rename(columns={"team": "opp", netc: "_m"})
    Gx = G[G.team.isin(rated.index)][["g", "team", "opp"] + [c for _, c in APEX_TRAITS if c in G]].copy()
    Gx = Gx.merge(mirror, on=["g", "opp"], how="left")
    so, co = Gx.opp.map(s["sum"]), Gx.opp.map(s["count"])
    loo = (so - Gx._m) / (co - 1).where(co > 1)
    Gx["opp_pct"] = pct_vs(T.loc[T.games >= POOL["min"], netc], loo).fillna(0.5).values
    Gx["w"] = np.where(Gx.opp_pct < 0.25, 0.5, 1.0)
    adjg = G[["g", "team"]].join(GA.drop(columns=["g", "team", "opp"]))
    Gx = Gx.merge(adjg, on=["g", "team"], how="left", suffixes=("", "_adj"))
    pct, cut = {}, {}
    for tier, grp in rated.groupby("tier"):
        base, _ = pools[tier]
        bg = Gx[Gx.team.isin(base.index)]
        for k, col in APEX_TRAITS:
            if col not in base or base[col].isna().all():
                continue
            pct.setdefault(k, {}).update(pct_vs(base[col], grp[col]).to_dict())
            if k == "runs":                                  # a game counts if it had a 12+ run
                cut.setdefault(k, {})[tier] = max(1.0, float(bg[col].quantile(APEX_CUT[k])))
            else:
                cut.setdefault(k, {})[tier] = float(bg[col + "_adj"].quantile(APEX_CUT[k]))
    out = {}
    for tid, r in rated.iterrows():
        games = Gx[Gx.team == tid]
        traits = {}
        for k, col in APEX_TRAITS:
            p = pct.get(k, {}).get(tid)
            c = cut.get(k, {}).get(r.tier)
            if p is None or pd.isna(p) or c is None or col not in games:
                traits[k] = None
                continue
            v = games[col] if k == "runs" else games[col + "_adj"]
            ok = v.notna()
            hit = (v >= c) & ok
            n, w_n = int(ok.sum()), float(games.w[ok].sum())
            w_hit = float(games.w[hit].sum())
            top = ok & (games.opp_pct >= 0.5)
            share = w_hit / w_n if w_n else 0.0
            solid = share * min(1.0, n / 5)                          # 1 of 2 games stays faint, 5 of 6 is solid
            traits[k] = dict(pct=round(float(p), 3), apex=bool(p >= APEX_CUT[k]), hit=int(hit.sum()), n=n,
                             top_hit=int((hit & top).sum()), top_n=int(top.sum()), solid=round(solid, 3))
        pil = [T[pp].rank(pct=True).get(tid) for pp in PILLARS if pp in T]
        parts = [x for x in pil if x is not None and not pd.isna(x)] + \
                [t["pct"] * (0.5 + 0.5 * t["solid"]) for t in traits.values() if t]   # consistency weighs every trait
        out[tid] = dict(traits=traits, core=sum(1 for t in traits.values() if t and t["apex"]),
                        score=round(sum(parts) / len(parts), 4) if parts else None)
    order = sorted((t for t in out if out[t]["score"] is not None), key=lambda t: -out[t]["score"])
    for i, t in enumerate(order):
        out[t]["rank"] = i + 1
    for t in out:
        out[t]["of"] = len(order)
    return out


def movers(teams, today, tiers=()):
    """Compare each team to its snapshot from about a week ago, plus last-5 pillar moves. A species change only counts
    when the snapshot was rated the same way (same method, pool size rule and tier pools)."""
    hist = sorted(f[:-5] for f in os.listdir(os.path.join(OUT, "history")) if f.endswith(".json"))
    old = [h for h in hist if h <= str(today - dt.timedelta(days=7))]
    prev = json.load(open(os.path.join(OUT, "history", old[-1] + ".json"))) if old else {}
    now_meta = dict(method=METHOD, pool_min=POOL["min"], tiers=sorted(tiers))
    same_method = prev.get("_meta") == now_meta
    out = []
    for tid, t in teams.items():
        if t["stage"] != "scored":
            continue
        then = (prev.get(tid) or {}).get("species") if same_method else None
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
    if old and not same_method:
        status["movers"] = f"species changes paused: the {old[-1]} snapshot was rated a different way (new method, pool size or tier pools). They resume once a matching snapshot is a week old"
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


def spread_check(details, hs, home_abbr, away_abbr):
    """ESPN's details text ('IONA -3.5') names the favorite. Returns the spread to use, or None when the two disagree."""
    d = str(details or "").strip()
    if hs is None or not d:
        return hs
    if d.upper() in ("EVEN", "PK", "PICK") or hs == 0:
        return hs
    abbr = d.rsplit(" ", 1)[0].strip().upper()
    if abbr == str(home_abbr).upper():
        return hs if hs < 0 else None
    if abbr == str(away_abbr).upper():
        return hs if hs > 0 else None
    return hs                                              # text we can't read: keep ESPN's number


LINE_DROPPED = []


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
        checked = spread_check(odds.get("details", ""), hs, h["team"].get("abbreviation", ""), a["team"].get("abbreviation", ""))
        if hs is not None and checked is None:
            LINE_DROPPED.append(f"{a['team'].get('displayName', '')} at {h['team'].get('displayName', '')} ({odds.get('details', '')}, spread {hs})")
        mlh, mla = ml_from(odds, "home"), ml_from(odds, "away")
        st = comp["status"]["type"]
        games.append(dict(id=str(ev["id"]), day=str(day), start=ev.get("date"), state=st.get("state"),
                          completed=bool(st.get("completed")) and st.get("name", "STATUS_FINAL") not in ("STATUS_POSTPONED", "STATUS_CANCELED", "STATUS_SUSPENDED"),
                          neutral=bool(comp.get("neutralSite")), home=str(h["team"]["id"]), away=str(a["team"]["id"]),
                          home_name=h["team"].get("displayName", ""), away_name=a["team"].get("displayName", ""),
                          details=odds.get("details", ""), home_spread=checked, ml_home=mlh if checked is not None or hs is None else None,
                          ml_away=mla if checked is not None or hs is None else None, line_dropped=hs is not None and checked is None,
                          home_score=num(h.get("score")), away_score=num(a.get("score"))))
    return games


def keep_pregame_lines(slate, path):
    """Once a game starts ESPN can drop or change the odds. Keep the last line we saw before tip-off."""
    try:
        old = {str(g["id"]): g for g in json.load(open(path)).get("games", [])}
    except Exception:
        return 0
    kept = 0
    for g in slate:
        o = old.get(g["id"])
        if not o or g["state"] == "pre" or o.get("home_spread") is None:
            continue
        for k in ("details", "home_spread", "ml_home", "ml_away", "line_dropped"):   # the saved record already holds the pregame line
            g[k] = o.get(k)
        kept += 1
    return kept


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


BOYDS_UPDATED = {"text": None, "date": None}
SCACC = "https://www.scacchoops.com/"
ALIASES = {"appalachian state": "app state",
           "connecticut": "uconn",
           "csun": "cal state northridge",
           "etsu": "east tennessee state",
           "fau": "florida atlantic",
           "fgcu": "florida gulf coast",
           "fiu": "florida international",
           "fsu": "florida state",
           "liu": "long island university",
           "louisiana lafayette": "louisiana",
           "miami (fl)": "miami",
           "mtsu": "middle tennessee",
           "pitt": "pittsburgh",
           "sfa": "stephen f. austin",
           "siue": "siu edwardsville",
           "sju": "saint joseph's",
           "smc": "saint mary's",
           "st. joseph's": "saint joseph's",
           "st. mary's": "saint mary's",
           "ucsb": "uc santa barbara",
           "umass": "massachusetts",
           "unc": "north carolina",
           "uncg": "unc greensboro",
           "uncw": "unc wilmington",
           "unf": "north florida",
           "usf": "south florida",
           "utrgv": "ut rio grande valley",
           "wku": "western kentucky",
           "arkansas little rock": "little rock",
           "arkansas pine bluff": "arkansas-pine bluff",
           "csub": "cal state bakersfield",
           "csuf": "cal state fullerton",
           "detroit": "detroit mercy",
           "fort wayne": "purdue fort wayne",
           "grambling state": "grambling",
           "illinois chicago": "uic",
           "illinois-chicago": "uic",
           "iu indy": "iu indianapolis",
           "iupui": "iu indianapolis",
           "lmu": "loyola marymount",
           "louisiana monroe": "ul monroe",
           "louisiana-monroe": "ul monroe",
           "loyola (chi)": "loyola chicago",
           "loyola (md)": "loyola maryland",
           "loyola-chicago": "loyola chicago",
           "mcneese state": "mcneese",
           "mississippi": "ole miss",
           "missouri kansas city": "kansas city",
           "nebraska omaha": "omaha",
           "nicholls state": "nicholls",
           "penn": "pennsylvania",
           "pfw": "purdue fort wayne",
           "queens": "queens university",
           "semo": "southeast missouri state",
           "southeast missouri": "southeast missouri state",
           "southeastern louisiana": "se louisiana",
           "southern mississippi": "southern miss",
           "tamucc": "texas a&m-corpus christi",
           "tennessee martin": "ut martin",
           "tennessee-martin": "ut martin",
           "texas a&m commerce": "east texas a&m",
           "texas a&m corpus christi": "texas a&m-corpus christi",
           "texas a&m-cc": "texas a&m-corpus christi",
           "texas a&m-commerce": "east texas a&m",
           "texas arlington": "ut arlington",
           "texas-arlington": "ut arlington",
           "ualr": "little rock",
           "uapb": "arkansas-pine bluff",
           "ulm": "ul monroe",
           "umkc": "kansas city",
           "ut-martin": "ut martin",
           "uta": "ut arlington",
           "southern california": "usc"}   # injury-site name -> ESPN short name, each checked against ESPN's team list


def load_injuries():
    """Boyd's Bets table. Also records when the page says it was last updated."""
    page = get(BOYDS, raw=True).decode("utf-8", "ignore")
    m = re.search(r"Injuries Last Updated on\s*(?:<[^>]+>\s*)*([A-Za-z]+,\s*[A-Za-z]+ \d{1,2},\s*\d{4})", page)
    if m:
        BOYDS_UPDATED["text"] = m.group(1)
        try:
            BOYDS_UPDATED["date"] = dt.datetime.strptime(re.sub(r"^[A-Za-z]+,\s*", "", m.group(1)), "%B %d, %Y").date()
        except Exception:
            pass
    tp = Table(); tp.feed(page)
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
        stt = re.split(r"\s*–\s*|\s+-\s*|\s*-\s+", st, maxsplit=1)   # 'Out - Knee' splits, 'Day-To-Day' stays whole
        out.append(dict(team=team, player=player, pos=pos, status=stt[0].strip(), injury=stt[1].strip() if len(stt) > 1 else "",
                        date=date, note=note, source="B", conf=None))
    if not out:
        raise ValueError("page loaded but no injury rows were found, layout may have changed")
    return out


def scacc_parse(page, conf):
    """SCACC lists a bold team header, then a Player / Injury / Status / Details table."""
    out, team, used = [], "", True
    for m in re.finditer(r"<(?:b|strong|h\d)[^>]*>(.*?)</(?:b|strong|h\d)>|<tr[^>]*>(.*?)</tr>", page, re.S | re.I):
        if m.group(1) is not None:
            t = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", m.group(1))).split())
            if t and len(t) < 40 and not t.lower().startswith(("player", "acc ", "select")):
                team, used = t, False
        else:
            cells = [" ".join(html.unescape(re.sub(r"<[^>]+>", " ", c)).split()) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", m.group(2), re.S | re.I)]
            if cells and cells[0].lower() == "player":
                if used:            # a second table with no new team header: don't hand its players to the last team
                    team = ""
                used = True
                continue
            if len(cells) >= 3 and re.search(r"Out|Questionable|Doubtful|Probable|Day", cells[2], re.I):
                if not team:
                    out.append(dict(team="(no team header)", player=cells[0], pos="", status=cells[2], injury="", date="", note="", source="S", conf=conf))
                    continue
                mm = re.match(r"(.*?)\s*\(([^)]*)\)\s*$", cells[0])
                player, pos = (mm.group(1), mm.group(2)) if mm else (cells[0], "")
                out.append(dict(team=team, player=player, pos=pos, status=cells[2], injury=cells[1], date="", note=cells[3] if len(cells) > 3 else "",
                                source="S", conf=conf))
    return out


def load_scacc():
    """Every conference page SCACC offers, loaded through the menu's own addresses."""
    from concurrent.futures import ThreadPoolExecutor
    home = get(SCACC + "acc-mens-basketball-injury-report", raw=True).decode("utf-8", "ignore")
    sel = re.search(r"<select[^>]*>(.*?)</select>", home, re.S | re.I)
    opts = [(v, t.strip()) for v, t in re.findall(r"<option[^>]*value=[\"']?([^\"'>]*)[\"']?[^>]*>([^<]*)", sel.group(1), re.I) if v] if sel else []
    if not opts:
        raise ValueError("conference menu not found, page layout may have changed")
    failed, errors = [], []

    def one(o):
        v, name = o
        try:
            return scacc_parse(get(urllib.parse.urljoin(SCACC, v), raw=True).decode("utf-8", "ignore"), name)
        except Exception as e:
            failed.append(name); errors.append(f"{type(e).__name__}: {e}"); return []
    with ThreadPoolExecutor(6) as ex:
        rows = [r for part in ex.map(one, opts) for r in part]
    ok = len(opts) - len(failed)
    SCACC_FAILED[:] = failed
    word = "ok" if not failed else "PARTIAL" if ok else "FAILED"
    status["SCACC"] = f"{word}: {len(rows)} listed across {ok} of {len(opts)} conferences" + (f", failed: {', '.join(failed[:6])}{'...' if len(failed) > 6 else ''}; first error: {errors[0][:160]}" if failed else "")
    if not ok:
        raise ValueError(errors[0] if errors else "no conference pages loaded")
    if not rows:
        raise ValueError(f"{ok} conference pages loaded but no injury rows were found, layout may have changed")
    return rows


SCACC_FAILED = []


SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?", re.I)


def plain(s):
    """strip accents so 'San José State' matches 'San Jose State'"""
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()


def norm(s):
    s = SUFFIX.sub("", plain(s).lower().replace("’", "").replace("'", ""))
    return re.sub(r"[^a-z0-9]", "", s)


def name_key(s):
    """last name + first initial, for matching 'A.J. Smith Jr.' to 'AJ Smith' or 'Cam Jones' to 'Cameron Jones'"""
    parts = [p for p in re.sub(r"[^a-z ]", "", SUFFIX.sub("", plain(s).lower())).split() if p]
    return (parts[-1], parts[0][0]) if len(parts) >= 2 else (norm(s), "")


def parse_date(s):
    s = re.sub(r"^[A-Za-z]{3,9},\s*", "", str(s).strip())          # Boyd's writes 'Sun, Nov 1'
    for fmt in ("%b %d", "%B %d", "%m/%d", "%m/%d/%Y", "%Y-%m-%d", "%b %d, %Y", "%B %d, %Y"):
        try:
            d = dt.datetime.strptime(s.replace(".", ""), fmt).date()
            if d.year == 1900:
                d = d.replace(year=TODAY.year if d.month <= TODAY.month + 1 else TODAY.year - 1)
            return d
        except Exception:
            pass
    return None


def team_finder(meta):
    names = {k: (v["name"], v["short"]) for k, v in D1_INFO.items()} or {k: (r["name"], r["short"]) for k, r in meta.iterrows()}
    by_name = {norm(n): k for k, (n, sh) in names.items()}
    by_short = {norm(sh): k for k, (n, sh) in names.items() if isinstance(sh, str)}
    # mascot = display name minus the school name ('UNLV Rebels' -> 'rebels'); used only to confirm a prefix match
    mascot = {k: norm(str(n)[len(str(sh)):]) if isinstance(sh, str) and str(n).startswith(str(sh)) else "" for k, (n, sh) in names.items()}

    def find(label, conf=None):
        low = " ".join(str(label).lower().split())
        if low == "miami":
            return by_short.get("miamioh") if conf and "mid-american" in conf.lower() else by_short.get("miami")
        low = ALIASES.get(low, low)
        for cand in (low, re.sub(r"(?<=\w) st\.?(?= |$)", " state", low)):   # 'Michigan St.', 'Boise St. Broncos' -> State, after trying the name as written
            cand = ALIASES.get(cand, cand)
            key = norm(cand)
            if key in by_name:
                return by_name[key]
            if key in by_short:
                return by_short[key]
        low = cand
        # 'UNLV Runnin' Rebels' vs ESPN's 'UNLV Rebels': school name at the front AND the label ends in that school's mascot.
        # No guessing on the school alone, so 'Michigan St.' can never land on Michigan.
        words = low.split()
        for i in range(len(words) - 1, 0, -1):
            tid = by_short.get(norm(" ".join(words[:i])))
            rest = norm(" ".join(words[i:]))
            if tid and mascot.get(tid) and (rest == mascot[tid] or rest.endswith(mascot[tid]) or mascot[tid].endswith(rest) and len(rest) >= 4):
                return tid
        return None
    return find


SEV = [("out", 4), ("doubt", 3), ("question", 2), ("day", 2), ("prob", 1)]


def severity(st):
    s = str(st).lower()
    return next((v for k, v in SEV if k in s), 2)


def rosters(P):
    """Scoring ranks (regulars only) plus who played in each team's last two games."""
    P = P.copy()
    P["team"] = P.team_id.astype(str)
    P["mins"] = pd.to_numeric(P.minutes, errors="coerce").fillna(0)
    P["pts"] = pd.to_numeric(P.points, errors="coerce").fillna(0)
    dcol = "game_date_time" if "game_date_time" in P else "game_date"
    pl = P[P.mins > 0]
    tg = pl.groupby(["team", "game_id"])[dcol].first().reset_index().sort_values(dcol)
    last2 = {t: set(g.game_id) for t, g in tg.groupby("team").tail(2).groupby("team")}
    last1 = {t: set(g.game_id) for t, g in tg.groupby("team").tail(1).groupby("team")}
    team_g = tg.groupby("team").game_id.nunique()
    agg = pl.groupby(["team", "athlete_display_name"]).agg(ppg=("pts", "mean"), gp=("pts", "size")).reset_index()
    by = pl.groupby(["team", "athlete_display_name"]).game_id.apply(set).to_dict()
    agg["l2"] = [len(by[(t, n)] & last2.get(t, set())) for t, n in zip(agg.team, agg.athlete_display_name)]
    agg["l1"] = [len(by[(t, n)] & last1.get(t, set())) for t, n in zip(agg.team, agg.athlete_display_name)]
    reg = agg[agg.gp >= (agg.team.map(team_g).fillna(1) / 2).clip(lower=1)].copy()
    reg["rank"] = reg.groupby("team").ppg.rank(ascending=False, method="first")
    return agg, reg


INJ_CARRY_DAYS = 7     # how long a player whose source page failed is kept on the board from his last listing
NEW_DAYS = 4           # a listing counts as new for this many days after we first see it (or after it gets worse)


def build_injuries(boyds, scacc, P, meta, box_stale=False, boyds_failed=False, scacc_failed=()):
    """Merge both sources, resolve teams and players, and flag which absences are new."""
    find = team_finder(meta)
    agg, reg = rosters(P)
    age = (TODAY - BOYDS_UPDATED["date"]).days if BOYDS_UPDATED["date"] else None
    boyds_stale = age is None or age > 2
    merged, unmatched = {}, []
    for i in list(scacc) + list(boyds):
        tid = find(i["team"], i.get("conf"))
        if tid is None:
            unmatched.append(i["team"]); continue
        key = (tid, norm(i["player"]))
        i = dict(i, tid=tid)
        if key not in merged:
            merged[key] = dict(i, sources={i["source"]}, boyds_date=i["date"] if i["source"] == "B" else "")
            continue
        m = merged[key]
        if i["source"] in m["sources"]:
            continue                                         # same player twice on one site: keep the first
        m["sources"].add(i["source"])
        if i["source"] == "B":
            m["boyds_date"] = i["date"]
            m["note"] = m.get("note") or i["note"]
            m["injury"] = m.get("injury") or i["injury"]
            if not boyds_stale and severity(i["status"]) > severity(m["status"]):
                m["status"] = i["status"]                    # fresh Boyd's and worse: be cautious
    # 'Cam Jones' on one site and 'Cameron Jones' on the other: merge across sources only, never within one
    path = os.path.join(OUT, "injury_state.json")
    old = json.load(open(path)) if os.path.exists(path) else None
    by_nk = {}
    for key, m in list(merged.items()):
        by_nk.setdefault((key[0], name_key(m["player"])), []).append(key)
    for keys in by_nk.values():
        s_only = [k for k in keys if merged[k]["sources"] == {"S"}]
        b_only = [k for k in keys if merged[k]["sources"] == {"B"}]
        if len(s_only) == 1 and len(b_only) == 1:
            ks, kb = s_only[0], b_only[0]
            keep, drop = (kb, ks) if f"{kb[0]}|{kb[1]}" in (old or {}) and f"{ks[0]}|{ks[1]}" not in (old or {}) else (ks, kb)
            m, b = merged[keep], merged.pop(drop)
            if b["source"] == "B" or "B" in b["sources"]:
                m["boyds_date"] = b.get("boyds_date") or b["date"]
            for f_ in ("note", "injury", "pos"):
                m[f_] = m.get(f_) or b.get(f_)
            if "B" in b["sources"] and not boyds_stale and severity(b["status"]) > severity(m["status"]):
                m["status"] = b["status"]
            m["sources"] |= b["sources"]
    # players whose source page failed this run: keep their last record for a while instead of dropping them
    failed_conf = set(scacc_failed)
    carried = 0
    for k, o in (old or {}).items():
        tid, pk = k.split("|", 1)
        if (tid, pk) in merged:
            continue
        src = o.get("src")
        if src is None:                                      # saved before sources were recorded
            lost = boyds_failed or "*" in failed_conf
        else:
            lost = ("S" in src and ("*" in failed_conf or o.get("conf") in failed_conf)) or ("B" in src and boyds_failed)
        if lost and (TODAY - dt.date.fromisoformat(o.get("last_seen", str(TODAY)))).days <= INJ_CARRY_DAYS:
            merged[(tid, pk)] = dict(carried=True, carried_state=o)
            carried += 1 if o.get("rec") else 0
    # first-seen tracking across daily runs
    state, out = {}, {}
    for (tid, pk), m in merged.items():
        k = f"{tid}|{pk}"
        if m.get("carried"):                                 # unchanged copy from the last good run; first-seen dates survive
            state[k] = m["carried_state"]
            if m["carried_state"].get("rec"):
                out.setdefault(tid, []).append(dict(m["carried_state"]["rec"], carried=True, fresh=False))
            continue
        o = (old or {}).get(k)
        sev_now = severity(m["status"])
        first = o["first_seen"] if o else ("before" if old is None else str(TODAY))
        changed = o.get("changed", first) if o else first
        if o and sev_now > o.get("sev", sev_now):
            changed = str(TODAY)                              # got worse, so it counts as new again
        bd = parse_date(m.get("boyds_date") or "")
        if bd and (first == "before" or (first != "before" and bd < dt.date.fromisoformat(first))):
            first = str(bd)
        when = max((dt.date.fromisoformat(x) for x in (first, changed) if x != "before"), default=None)
        listed_recent = when is not None and (TODAY - when).days <= NEW_DAYS
        # one matched player for every field: exact name first, then last name + first initial only if it is unique
        sub = agg[agg.team == tid]
        row = sub[[norm(n) == pk for n in sub.athlete_display_name]]
        if row.empty:
            row = sub[[name_key(n) == name_key(m["player"]) for n in sub.athlete_display_name]]
            if len(row) > 1:
                row = row.iloc[0:0]                           # two candidates (brothers, twins): don't guess
        ath = row.athlete_display_name.iloc[0] if len(row) else None
        rk = reg[(reg.team == tid) & (reg.athlete_display_name == ath)] if ath else reg.iloc[0:0]
        gp = int(row.gp.iloc[0]) if len(row) else None
        played_last = bool(row.l1.iloc[0]) if len(row) else None
        missed = bool(len(row) and gp >= 3 and row.l2.iloc[0] == 0)
        rank = int(rk["rank"].iloc[0]) if len(rk) else None
        ppg = round(float(row.ppg.iloc[0]), 1) if len(row) else None
        text = f"{m.get('note', '')} {m.get('injury', '')} {m['status']}".lower()
        long = bool(re.search(r"season|indefinite|acl|achilles|redshirt|surgery|months", text))
        src = "".join(sorted(m["sources"]))
        # new absence = listed recently AND he played his team's last game (or we can't find him in the box scores).
        # A leftover listing for a player who is back fails the first test; stale box scores turn the flag off.
        fresh = bool(listed_recent and (played_last is not False)) and not box_stale
        rec = dict(player=m["player"], pos=m.get("pos", ""), status=m["status"].split()[0].title() if m["status"] else "",
                   injury=m.get("injury", ""), source=src, first_seen=first, fresh=fresh,
                   missed=missed, played_last=played_last, long=long, rank=rank, ppg=ppg, top2=bool(rank and rank <= 2),
                   stale=bool(src == "B" and boyds_stale), sev=sev_now)
        out.setdefault(tid, []).append(rec)
        state[k] = dict(first_seen=first, changed=changed, sev=sev_now, status=m["status"], src=src, conf=m.get("conf"),
                        last_seen=str(TODAY), rec=rec)
    os.makedirs(OUT, exist_ok=True)
    json.dump(state, open(path, "w"), separators=(",", ":"))
    for v in out.values():
        v.sort(key=lambda x: (not x["top2"], -x["sev"], x["rank"] or 99))
    live = [m for m in merged.values() if not m.get("carried")]
    b = f"Boyd's {sum(1 for m in live if 'B' in m['sources'])} players, last updated {BOYDS_UPDATED['text'] or 'unknown'}" + (f" ({age} days old, so SCACC wins any disagreement)" if boyds_stale and age is not None else "")
    status["injury sources"] = f"{b}; SCACC {sum(1 for m in live if 'S' in m['sources'])} players; {len(live)} unique, {sum(len(v) for v in out.values()) - carried} matched to teams" + (f"; {carried} kept from the last run because their page failed" if carried else "")
    if not (boyds or scacc):
        status["injury sources"] = f"both sources failed; {carried} players kept from the last good list, first-seen dates preserved"
    if unmatched:
        status["injury teams not matched"] = ", ".join(sorted(set(unmatched)))[:400]
    status["injury_meta"] = dict(boyds_updated=BOYDS_UPDATED["text"], boyds_age_days=age, scacc_ok="SCACC" in status and str(status["SCACC"]).startswith("ok"),
                                 carried=carried, box_stale=box_stale)
    return out


def hexcolor(c):
    c = str(c).lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    return "#" + c[:6] if re.fullmatch(r"[0-9a-fA-F]{6}", c[:6] or "") else "#888888"


# ---------------------------------------------------------------- assemble
def box_age(G, finals):
    """Days between the last box score game and today, counting only when real games were played after it
    (so the Christmas break doesn't read as stale data)."""
    last = dt.date.fromisoformat(str(G.date.max())[:10])
    missed = sorted({f["date"] for f in finals.values() if last < dt.date.fromisoformat(f["date"][:10]) < TODAY})
    age = (TODAY - last).days
    return last, age, bool(missed and age >= 2)


def main():
    os.makedirs(os.path.join(OUT, "history"), exist_ok=True)
    T, P, meta, d1, t, G = None, None, None, None, None, None
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
        T = None
    slate, seen = [], set()
    for k in range(3):
        day = TODAY + dt.timedelta(days=k)
        try:
            for g in load_slate(day):
                if g["id"] not in seen:                      # a late game can show up under two dates
                    seen.add(g["id"]); slate.append(g)
        except Exception as e:
            status[f"slate {day}"] = f"failed: {e}"
    kept = keep_pregame_lines(slate, os.path.join(OUT, "slate.json"))
    status["slate"] = f"{len(slate)} games over 3 days, {sum(g['home_spread'] is not None for g in slate)} with spreads" + (f", {kept} started games keep their pregame line" if kept else "")
    if LINE_DROPPED:
        status["lines dropped"] = f"{len(LINE_DROPPED)} spread(s) hidden because ESPN's number and favorite disagreed: " + "; ".join(LINE_DROPPED[:4])
    # final scores for grading picks, merged into what earlier runs saved so a failed pull never erases a result.
    # Only games ESPN marks completed count: postponed and canceled games also show as 'post'.
    path = os.path.join(OUT, "results.json")
    try:
        finals = json.load(open(path)).get("games", {})
    except Exception:
        finals = {}
    finals = {k: v for k, v in finals.items() if (v.get("home_score") or 0) + (v.get("away_score") or 0) > 0}
    before = len(finals)
    espn = []
    try:
        espn = load_slate(TODAY - dt.timedelta(days=1))
    except Exception as e:
        status["results (ESPN yesterday)"] = f"failed: {e}"
    for g in espn + slate:
        if g["state"] == "post" and not g.get("completed"):
            finals.pop(g["id"], None)                         # postponed or canceled: never grade it
        elif g.get("completed") and g["home_score"] is not None and g["away_score"] is not None and g["home_score"] + g["away_score"] > 0:
            finals[g["id"]] = dict(date=g["day"], home=g["home"], away=g["away"], home_score=g["home_score"], away_score=g["away_score"])
    try:
        if t is None:
            raise ValueError("no box scores this run")
        tb = t[["game_id", "game_date", "team_id", "team_home_away", "team_score"]].copy()
        tb["game_id"] = tb.game_id.astype(str); tb["team_id"] = tb.team_id.astype(str)
        cutoff = str(TODAY - dt.timedelta(days=21))
        for gid, x in tb[tb.game_date.astype(str) >= cutoff].groupby("game_id"):
            h, a = x[x.team_home_away == "home"], x[x.team_home_away == "away"]
            if len(h) == 1 and len(a) == 1 and gid not in finals:
                hs_, as_ = float(h.team_score.iloc[0]), float(a.team_score.iloc[0])
                if hs_ > 0 and as_ > 0:
                    finals[gid] = dict(date=str(h.game_date.iloc[0])[:10], home=h.team_id.iloc[0], away=a.team_id.iloc[0], home_score=hs_, away_score=as_)
    except Exception as e:
        status["results (box scores)"] = f"failed: {e}"
    status["results"] = f"{len(finals)} finals saved ({len(finals) - before:+d} this run)"
    box_stale = False
    if G is not None:
        last, age, box_stale = box_age(G, finals)
        if box_stale:
            status["box scores"] += f" (WARNING: {age} days old; 'new absence' flags are off until it catches up)"
    inj, boyds, scacc = {}, [], []
    boyds_failed, scacc_failed = False, []
    try:
        boyds = load_injuries()
    except Exception as e:
        status["Boyd's"] = f"failed: {e}"; boyds_failed = True
    try:
        scacc = load_scacc()
        scacc_failed = list(SCACC_FAILED)
    except Exception as e:
        status["SCACC"] = f"failed: {e}"; scacc_failed = ["*"]
    if T is not None and (boyds or scacc or os.path.exists(os.path.join(OUT, "injury_state.json"))):
        try:
            inj = build_injuries(boyds, scacc, P, meta, box_stale=box_stale, boyds_failed=boyds_failed, scacc_failed=scacc_failed)
        except Exception as e:
            status["injuries"] = f"failed: {e}"

    teams, crit, apx = {}, {}, {}
    if T is not None:
        pool_state_path = os.path.join(OUT, "pool_state.json")
        try:
            pool_state = json.load(open(pool_state_path))
        except Exception:
            pool_state = {}
        try:
            crit = critters(T, pool_state)
        except Exception as e:
            status["critters"] = f"failed: {e}"
        try:
            apx = apex_profile(T, G, pool_state)
        except Exception as e:
            status["apex"] = f"failed: {e}"
        json.dump(pool_state, open(pool_state_path, "w"))
        status["pools"] = f"rated against teams with {POOL['min']}+ games ({POOL['n']} teams); tiers on their own pool: {', '.join(sorted(pool_state.get('tiers_locked', {}))) or 'none yet'}"
        if ZERO_FLAGS:
            status["box zeros"] = f"{sum(z['blanked'] for z in ZERO_FLAGS)} stat(s) blanked as implausible, {sum(not z['blanked'] for z in ZERO_FLAGS)} kept but worth a look"
        f = lambda x, d=1: None if x is None or pd.isna(x) else round(float(x), d)
        for tid, r in T.iterrows():
            c = crit.get(tid, {})
            teams[tid] = dict(name=r["name"], short=r["short"], color=hexcolor(r["color"]), conf=None if pd.isna(r.get("conf")) else int(r["conf"]),
                              games=int(r.games), record=f"{int(r.wins)}-{int(r.games - r.wins)}", margin=f(r.margin),
                              stage=r.stage, big_win=bool(r.big_win), ato=f(r.ato_raw, 2), efg_z=f(r["eFG z"], 2), orbtv=f(r["ORBTV per 100"]),
                              last5=r.get("last5"), last5_wins=None if pd.isna(r.get("last5_wins")) else int(r["last5_wins"]),
                              efg_move=f(r.get("eFG move"), 2), fin=f(r.get("fin_rate"), 2), fin_n=None if pd.isna(r.get("fin_chances", float("nan"))) else int(r["fin_chances"]),
                              pillars={p: f(r[p], 2) for p in PILLARS}, moves={p: f(r[p + " move"], 2) for p in PILLARS},
                              species=c.get("sp", "Blob"), critter=c or None, apex=apx.get(tid))
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
    if slate or not os.path.exists(os.path.join(OUT, "slate.json")):
        json.dump(dict(date=str(TODAY), updated=stamp, games=out_games), open(os.path.join(OUT, "slate.json"), "w"))
    json.dump(dict(updated=stamp, games=finals), open(path, "w"), separators=(",", ":"))
    if teams:                                                 # a failed box score pull keeps yesterday's team files as they were
        mv = movers(teams, TODAY, pool_state.get("tiers_locked", {}))
        json.dump(dict(updated=stamp, teams=teams), open(os.path.join(OUT, "teams.json"), "w"), separators=(",", ":"))
        json.dump(dict(updated=stamp, movers=mv), open(os.path.join(OUT, "movers.json"), "w"))
        snap = {k: dict(games=v["games"], record=v["record"], stage=v["stage"], species=v["species"], pillars=v["pillars"],
                        efg_z=v["efg_z"], margin=v["margin"]) for k, v in teams.items()}
        snap["_meta"] = dict(method=METHOD, pool_min=POOL["min"], tiers=sorted(pool_state.get("tiers_locked", {})))
        json.dump(snap, open(os.path.join(OUT, "history", f"{TODAY}.json"), "w"), separators=(",", ":"))
    else:
        status["team files"] = "not updated this run (no box scores); the board shows the last good data"
    json.dump(dict(updated=stamp, date=str(TODAY), status=status, zero_flags=ZERO_FLAGS[:200]), open(os.path.join(OUT, "latest.json"), "w"))
    for k, v in status.items():
        print(f"{k:14s} {v}")
    if T is None and not slate:
        sys.exit(1)


if __name__ == "__main__":
    main()
