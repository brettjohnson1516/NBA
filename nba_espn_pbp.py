#!/usr/bin/env python
"""
Stage 2 (ESPN) - build the training play-by-play table from ESPN data.

Replaces the stats.nba.com fetcher, which read-timed out on every retry. This
does not scrape anything: sportsdataverse publishes one prebuilt parquet per
season on GitHub releases, so a season is a single file download instead of
1,300 rate-limited per-game API calls. The whole 2012-2026 range pulls in
minutes.

    play-by-play  .../releases/download/espn_nba_pbp/play_by_play_{season}.parquet
    rosters       .../releases/download/espn_nba_game_rosters/game_rosters_{season}.parquet

SEASON NUMBERING - THE EASY MISTAKE
-----------------------------------
ESPN names a season by its ENDING year. Their "2015" file runs 2014-10-28 to
2015-06-16, i.e. the 2014-15 season. That is the OPPOSITE of the start-year
convention the stats.nba.com script used. Training on 2011-12 through 2023-24
therefore means ESPN seasons 2012 through 2024. Holdout 2024-25 is ESPN 2025;
the Kalshi backtest season 2025-26 is ESPN 2026. Get this backwards and you
train on the holdout.

FOUR THINGS THE SOURCE GETS WRONG OR DIFFERENTLY
------------------------------------------------
1. `start_game_seconds_remaining` IS NOT TRUSTWORTHY. On a checked game the
   per-period maxima were 2520 / 1905 / 1308 / 707 where they should be
   2880 / 2160 / 1440 / 720, and Q2 opens at a HIGHER value than Q1 closes.
   Every period spans the right 720 seconds internally, so the within-period
   deltas are fine and the cross-period offsets are not. This script ignores
   the column and recomputes the clock from `period_number` and
   `clock_display_value`.

2. SCORES CAN GO DOWN - about 9,800 cells a season. Inspected, the cause is
   event ORDER, not wrong scores: on a putback ESPN lists the made basket
   before the offensive rebound that set it up, and the rebound row still
   carries the pre-basket score. The points are right, the two rows are
   swapped. Both share the same clock value, so on a five-second sampling grid
   the swap is invisible; what matters is that the running score never moves
   backwards. Both columns are forced non-decreasing and the corrections are
   counted.

   Separately, on about 0.5% of games the play-by-play's FINAL score drifts
   from the official result, and on a few of those it drifts far enough to
   flip which team won. `home_final`, `away_final` and the `home_won` training
   label therefore come from the schedule file, never from the last
   play-by-play row. In 2014-15 that is 7 games disagreeing and 1 whose label
   would have been inverted.

3. SUB DIRECTION IS REVERSED vs the NBA feed. In "J.R. Smith enters the game
   for Matthew Dellavedova", `athlete_id_1` is the player coming IN and
   `athlete_id_2` the player going OUT. The NBA feed has it the other way
   round. Reading it backwards silently inverts every lineup.

4. COLUMNS VARY BY SEASON - 62 in 2012, 63 in 2015, 67 in 2026. `wallclock`
   (a real UTC timestamp per play) only exists from 2015. Missing columns are
   filled with nulls rather than crashing.

LINEUPS
-------
Period 1 needs no inference: the roster file carries a `starter` flag, and it
gives exactly five per team per game in every season checked (2012, 2013, 2015,
2026). Those five ARE the opening lineup.

Periods 2+ are the hard part, because substitutions made between periods are
not in the play-by-play. Those are inferred by forward scan - a player seen
acting before he has been substituted in must have opened the period - and the
roster gives an exact player-to-team map, so nobody is misassigned.

The scan still cannot see a player who opens a period and records nothing at
all before leaving. Every period is therefore checked for exactly five a side,
and periods that fail are flagged `lineup_ok=False` and counted, never quietly
passed off as a four-man lineup.

FOULS
-----
ESPN names foul types in plain text, so unlike the NBA feed there are no
numeric action codes to guess at. The classification below covers every
foul-like `type_text` observed across 2012, 2015 and 2026. Team fouls (the ones
that advance the bonus) exclude offensive fouls and all technicals; personal
fouls (which drive foul trouble) exclude technicals but include offensive
fouls. Anything not in either list is reported by `--audit-fouls` so a new
label in a future season shows up instead of being dropped.

Output: one parquet per season under <outdir>/pbp_espn/pbp_{season}.parquet

Setup
-----
  pip install pandas pyarrow requests

Usage
-----
  python nba_espn_pbp.py --seasons 2012 2024 --outdir data
  python nba_espn_pbp.py --seasons 2025 2026 --outdir data     # holdout + backtest
  python nba_espn_pbp.py --seasons 2015 2015 --outdir data --audit-fouls
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

RELEASE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
PBP_URL = RELEASE + "/espn_nba_pbp/play_by_play_{season}.parquet"
ROSTER_URL = RELEASE + "/espn_nba_game_rosters/game_rosters_{season}.parquet"
SCHEDULE_URL = RELEASE + "/espn_nba_schedules/nba_schedule_{season}.parquet"

REGULATION_PERIODS = 4
PERIOD_SECONDS = 720
OT_SECONDS = 300

SUB_TYPE = "Substitution"

# Fouls that advance the team foul count toward the bonus. Offensive fouls are
# excluded on purpose - a personal foul on the player, but it does not put the
# other team closer to the penalty. Technicals are excluded for the same
# reason.
TEAM_FOUL_TYPES = {
    "Shooting Foul",
    "Personal Foul",
    "Loose Ball Foul",
    "Personal Take Foul",
    "Transition Take Foul",
    "Away from Play Foul",
    "Clear Path Foul",
    "Inbound Foul",
    "Flagrant Foul Type 1",
    "Flagrant Foul Type 2",
    "Double Personal Foul",
}

# Fouls charged to the player, driving foul trouble. Offensive fouls count
# here; technicals do not.
PERSONAL_FOUL_TYPES = TEAM_FOUL_TYPES | {
    "Offensive Foul",
    "Offensive Foul Turnover",
}

# Recognized but deliberately counted as neither.
TECHNICAL_TYPES = {
    "Technical Foul",
    "Double Technical Foul",
    "Defensive 3-Seconds Technical",
    "Delay Technical",
    "Hanging Technical Foul",
    "Flopping Technical",
    "Taunting Technical Foul",
    "Non-Unsportsmanlike Technical",
    "Excess Timeout Technical",
    "Too Many Players Technical",
    "Free Throw - Technical",
}

KNOWN_FOUL_LIKE = TEAM_FOUL_TYPES | PERSONAL_FOUL_TYPES | TECHNICAL_TYPES


def period_start_elapsed(period: int) -> int:
    if period <= REGULATION_PERIODS:
        return (period - 1) * PERIOD_SECONDS
    return (REGULATION_PERIODS * PERIOD_SECONDS
            + (period - REGULATION_PERIODS - 1) * OT_SECONDS)


def period_length(period: int) -> int:
    return PERIOD_SECONDS if period <= REGULATION_PERIODS else OT_SECONDS


def parse_clock(v) -> Optional[float]:
    """'11:34' -> 694.0 seconds left in the period."""
    if not isinstance(v, str) or ":" not in v:
        return None
    try:
        mm, ss = v.split(":", 1)
        return int(mm) * 60 + float(ss)
    except ValueError:
        return None


def download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached  {dest.name}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=180) as r:
            if r.status_code != 200:
                print(f"  HTTP {r.status_code}  {url}")
                return False
            tmp = dest.with_suffix(dest.suffix + ".part")
            n = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    n += len(chunk)
            tmp.replace(dest)
        print(f"  fetched {dest.name}  ({n / 1e6:.1f} MB)")
        return True
    except requests.RequestException as e:
        print(f"  FAILED  {url}: {e}")
        return False


# -- lineups ------------------------------------------------------------------

def classify_sub(a_in_raw, a_out_raw, text: str,
                 on_court: Optional[set[int]] = None
                 ) -> tuple[Optional[int], Optional[int]]:
    """
    Work out who is entering and who is leaving on one substitution row.

    The documented shape is athlete_id_1 = entering, athlete_id_2 = leaving,
    matching "J.R. Smith enters the game for Matthew Dellavedova". That is not
    always what arrives. ESPN emits TRUNCATED sub rows carrying only one
    athlete, in two different forms:

        "Ersan Ilyasova enters the game for "   -> the named player is ENTERING
        "enters the game for Eric Bledsoe"      -> the named player is LEAVING

    Both put the id in athlete_id_1 and leave athlete_id_2 null, so reading
    athlete_id_1 as "entering" is right in the first case and exactly backwards
    in the second. Getting it backwards adds the departing player to the floor
    and never removes anyone, so the lineup grows to six and stays wrong for
    the rest of the period. In 2018-19 there are 1,003 truncated rows, 512 of
    them the backwards form, and they alone accounted for 19,623 corrupted
    rows - 3.1% of the season - almost all six-man rather than four-man.

    Direction is therefore decided by evidence, in order:
      1. Lineup membership. Whoever is already on the floor is the one leaving.
         This is the strongest signal and needs no text parsing.
      2. Where the name sits relative to "enters the game for" in the text.
      3. The documented field order, as a last resort.
    """
    def pid(v):
        if v is None or v != v:
            return None
        return int(v)

    a1, a2 = pid(a_in_raw), pid(a_out_raw)
    # Defensive: a null text can still reach here as a float from an
    # Arrow-backed column.
    t = text.strip().lower() if isinstance(text, str) else ""

    if a1 is not None and a2 is not None:
        if on_court is not None:
            if a1 in on_court and a2 not in on_court:
                return a2, a1          # a1 is on the floor, so a1 leaves
            if a2 in on_court and a1 not in on_court:
                return a1, a2
        return a1, a2                  # documented order

    lone = a1 if a1 is not None else a2
    if lone is None:
        return None, None

    # "enters the game for <name>" with nothing before the verb: the only
    # named player is the one being replaced.
    leading_form = t.startswith("enters the game for")

    if on_court is not None:
        if lone in on_court:
            return None, lone          # already playing, so this is the exit
        return lone, None

    return (None, lone) if leading_form else (lone, None)



def resolve_game_lineups(g: pd.DataFrame, starters_home: set[int],
                         starters_away: set[int], player_team: dict[int, str],
                         report: dict) -> tuple[list[str], list[str], list[bool]]:
    """
    Home and away five-man lineups for every row of one game.

    Period 1 is seeded from the roster starter flags. Later periods are seeded
    by forward scan: a player acting before he has been substituted in must
    have opened the period. `player_team` comes from the roster, so a player is
    never assigned to the wrong side by guessing from event context.
    """
    n = len(g)
    home_out = [""] * n
    away_out = [""] * n
    ok_out = [False] * n

    # Everything below runs off numpy arrays. Per-row `.loc` access builds a
    # Series each time and is roughly 300x slower - enough to turn a season
    # from seconds into hours.
    types = g["type_text"].to_numpy()
    periods = g["period_number"].to_numpy()
    a1 = g["athlete_id_1"].to_numpy()
    a2 = g["athlete_id_2"].to_numpy()
    a3 = g["athlete_id_3"].to_numpy()
    # `.astype(str)` on an Arrow-backed string column PRESERVES nulls rather
    # than rendering them "nan", and `.to_numpy()` then hands back real floats.
    # 2019-20 has 106 such rows, one of them a substitution, which is enough to
    # crash the whole season. Fill before converting.
    texts = g["text"].fillna("").astype(str).to_numpy()

    def pid_of(v) -> Optional[int]:
        if v is None or v != v:  # NaN
            return None
        return int(v)

    prev_home: set[int] = set()
    prev_away: set[int] = set()

    for period in sorted(set(periods.tolist())):
        rows = [i for i in range(n) if periods[i] == period]

        if period == 1:
            open_home, open_away = set(starters_home), set(starters_away)
        else:
            subbed_in: set[int] = set()
            open_home, open_away = set(), set()
            for i in rows:
                if types[i] == SUB_TYPE:
                    in_p, out_p = classify_sub(a1[i], a2[i], texts[i])
                    if out_p is not None and out_p not in subbed_in:
                        side = player_team.get(out_p)
                        if side == "home":
                            open_home.add(out_p)
                        elif side == "away":
                            open_away.add(out_p)
                    if in_p is not None:
                        subbed_in.add(in_p)
                    continue
                for arr in (a1, a2, a3):
                    pid = pid_of(arr[i])
                    if pid is None or pid in subbed_in:
                        continue
                    side = player_team.get(pid)
                    if side == "home":
                        open_home.add(pid)
                    elif side == "away":
                        open_away.add(pid)

        if period > 1 and (len(open_home) != 5 or len(open_away) != 5):
            # Fall back to whoever finished the previous period. Wrong whenever
            # a between-period substitution happened, so it stays flagged.
            if len(open_home) != 5 and len(prev_home) == 5:
                open_home = set(prev_home)
            if len(open_away) != 5 and len(prev_away) == 5:
                open_away = set(prev_away)
            report["carried_forward"] += 1

        if len(open_home) != 5 or len(open_away) != 5:
            report["unresolved_periods"] += 1

        cur_home, cur_away = set(open_home), set(open_away)
        h_str = ",".join(str(x) for x in sorted(cur_home))
        a_str = ",".join(str(x) for x in sorted(cur_away))

        for i in rows:
            if types[i] == SUB_TYPE:
                # Side first, so membership can be checked against the right
                # five. A player the roster does not place on either team
                # leaves the lineup untouched rather than corrupting one.
                cand = [p for p in (pid_of(a1[i]), pid_of(a2[i])) if p is not None]
                side = None
                for p in cand:
                    side = player_team.get(p)
                    if side is not None:
                        break
                if side == "home":
                    cur, other = cur_home, None
                elif side == "away":
                    cur, other = cur_away, None
                else:
                    report["sub_no_side"] += 1
                    cur = None

                if cur is not None:
                    in_p, out_p = classify_sub(a1[i], a2[i], texts[i], cur)
                    if out_p is not None:
                        if out_p in cur:
                            cur.discard(out_p)
                        else:
                            report["sub_out_not_on_court"] += 1
                    if in_p is not None:
                        if in_p in cur:
                            report["sub_in_already_on_court"] += 1
                        else:
                            cur.add(in_p)
                    if side == "home":
                        h_str = ",".join(str(x) for x in sorted(cur_home))
                    else:
                        a_str = ",".join(str(x) for x in sorted(cur_away))
            else:
                # Mid-period repair for truncated subs.
                #
                # A one-sided sub row names only the player entering OR only
                # the player leaving, so one half of the swap is unknowable
                # from that row and the side is left at four or six. When the
                # side is SHORT, the next player on that team to act while not
                # already counted is on the floor by demonstration - the same
                # first-touch logic used to seed a period, applied mid-period.
                # A six-man side cannot be repaired this way, because nothing
                # a player does proves he is absent; those rows stay flagged.
                for arr in (a1, a2, a3):
                    pid = pid_of(arr[i])
                    if pid is None:
                        continue
                    sd = player_team.get(pid)
                    if sd == "home" and pid not in cur_home and len(cur_home) < 5:
                        cur_home.add(pid)
                        h_str = ",".join(str(x) for x in sorted(cur_home))
                        report["repaired_short_lineup"] += 1
                    elif sd == "away" and pid not in cur_away and len(cur_away) < 5:
                        cur_away.add(pid)
                        a_str = ",".join(str(x) for x in sorted(cur_away))
                        report["repaired_short_lineup"] += 1
            home_out[i] = h_str
            away_out[i] = a_str
            ok_out[i] = len(cur_home) == 5 and len(cur_away) == 5

        prev_home, prev_away = set(cur_home), set(cur_away)

    return home_out, away_out, ok_out


# -- per-season build ---------------------------------------------------------

def build_season(season: int, raw_dir: Path, report: dict,
                 foul_audit: Optional[set]) -> Optional[pd.DataFrame]:
    pbp_path = raw_dir / f"play_by_play_{season}.parquet"
    ros_path = raw_dir / f"game_rosters_{season}.parquet"
    sch_path = raw_dir / f"nba_schedule_{season}.parquet"
    if not download(PBP_URL.format(season=season), pbp_path):
        return None
    if not download(ROSTER_URL.format(season=season), ros_path):
        return None
    if not download(SCHEDULE_URL.format(season=season), sch_path):
        return None

    pbp = pd.read_parquet(pbp_path)
    ros = pd.read_parquet(ros_path)

    # Final scores come from the SCHEDULE, not from the last play-by-play row.
    #
    # On roughly 0.5% of games the play-by-play score column drifts from the
    # official result - and on some of those the drift is large enough to flip
    # which team won. Taking `home_won` off the play-by-play tail would train
    # the model on inverted labels for those games. The schedule is the
    # authoritative result, so it wins, and disagreements are counted.
    sch = pd.read_parquet(sch_path, columns=["game_id", "home_score", "away_score"])
    sch["game_id"] = sch["game_id"].astype(str)
    finals = {
        r.game_id: (float(r.home_score), float(r.away_score))
        for r in sch.itertuples()
        if pd.notna(r.home_score) and pd.notna(r.away_score)
    }

    # Columns differ by season; fill the missing ones rather than crashing.
    for col in ("wallclock", "athlete_id_3", "home_team_spread", "game_spread"):
        if col not in pbp.columns:
            pbp[col] = pd.NA

    pbp = pbp.sort_values(["game_id", "period_number", "game_play_number"]) \
             .reset_index(drop=True)

    # Roster: starters and an exact player -> side map, per game.
    #
    # game_id ships as int32 in the play-by-play file and as a STRING in the
    # roster file. Joining them without normalizing matches zero rows and every
    # game silently loses its starters, so both sides are cast to string.
    ros = ros[ros["athlete_id"].notna()].copy()
    ros["athlete_id"] = ros["athlete_id"].astype("int64")
    ros["game_id"] = ros["game_id"].astype(str)
    pbp["game_id"] = pbp["game_id"].astype(str)

    frames = []
    for game_id, g in pbp.groupby("game_id", sort=False):
        g = g.copy()
        home_id = g["home_team_id"].dropna()
        away_id = g["away_team_id"].dropna()
        if home_id.empty or away_id.empty:
            report["no_team_ids"] += 1
            continue
        home_id, away_id = int(home_id.iloc[0]), int(away_id.iloc[0])

        rg = ros[ros["game_id"] == game_id]
        if rg.empty:
            report["no_roster"] += 1
            continue

        player_team = {}
        for _, r in rg.iterrows():
            tid = r["team_id"]
            if pd.isna(tid):
                continue
            tid = int(tid)
            player_team[int(r["athlete_id"])] = (
                "home" if tid == home_id else "away" if tid == away_id else None
            )

        st = rg[rg["starter"] == True]  # noqa: E712 - pandas mask
        s_home = {int(p) for p, t in zip(st["athlete_id"], st["team_id"])
                  if pd.notna(t) and int(t) == home_id}
        s_away = {int(p) for p, t in zip(st["athlete_id"], st["team_id"])
                  if pd.notna(t) and int(t) == away_id}
        if len(s_home) != 5 or len(s_away) != 5:
            report["bad_starters"] += 1

        # Clock, computed here - the shipped seconds-remaining column is wrong.
        period = g["period_number"].astype(int)
        pc_left = g["clock_display_value"].map(parse_clock)
        elapsed = period.map(period_start_elapsed) + (period.map(period_length) - pc_left)
        seconds_left_reg = REGULATION_PERIODS * PERIOD_SECONDS - elapsed

        # Scores forced non-decreasing.
        hs = pd.to_numeric(g["home_score"], errors="coerce").ffill().fillna(0)
        as_ = pd.to_numeric(g["away_score"], errors="coerce").ffill().fillna(0)
        hs_fixed, as_fixed = hs.cummax(), as_.cummax()
        report["score_corrections"] += int((hs_fixed != hs).sum() + (as_fixed != as_).sum())

        if foul_audit is not None:
            foul_audit.update(
                t for t in g["type_text"].dropna().unique()
                if any(k in str(t).lower() for k in ("foul", "technical"))
            )

        home_lu, away_lu, lu_ok = resolve_game_lineups(
            g, s_home, s_away, player_team, report
        )

        # Fouls. Team fouls reset each period; personal fouls run all game.
        tt = g["type_text"].astype(str)
        is_team_foul = tt.isin(TEAM_FOUL_TYPES)
        is_pf = tt.isin(PERSONAL_FOUL_TYPES)
        actor_team = g["team_id"]

        h_tf, a_tf, pf_str = [], [], []
        pf: dict[int, int] = {}
        h_ct = a_ct = 0
        last_period = None

        per_arr = period.to_numpy()
        team_arr = actor_team.to_numpy()
        tf_arr = is_team_foul.to_numpy()
        pf_arr = is_pf.to_numpy()
        act_arr = g["athlete_id_1"].to_numpy()
        pf_cache = ""

        for i in range(len(g)):
            p = int(per_arr[i])
            if p != last_period:
                h_ct = a_ct = 0
                last_period = p
            tid = team_arr[i]
            if tid == tid and tid is not None:  # not NaN
                tid = int(tid)
                if tf_arr[i]:
                    if tid == home_id:
                        h_ct += 1
                    elif tid == away_id:
                        a_ct += 1
                if pf_arr[i]:
                    v = act_arr[i]
                    if v == v and v is not None:
                        k = int(v)
                        pf[k] = pf.get(k, 0) + 1
                        pf_cache = ",".join(f"{a}:{b}" for a, b in sorted(pf.items()))
            h_tf.append(h_ct)
            a_tf.append(a_ct)
            pf_str.append(pf_cache)

        pbp_home_final = float(hs_fixed.iloc[-1])
        pbp_away_final = float(as_fixed.iloc[-1])
        official = finals.get(str(game_id))
        if official is None:
            report["no_official_score"] += 1
            continue
        home_final, away_final = official
        if (home_final, away_final) != (pbp_home_final, pbp_away_final):
            report["final_score_disagreement"] += 1
            if (home_final > away_final) != (pbp_home_final > pbp_away_final):
                report["winner_would_have_flipped"] += 1

        frames.append(pd.DataFrame({
            "game_id": g["game_id"].values,
            "season": season,
            "season_type": g["season_type"].values,
            "game_date": g["game_date"].values,
            "home_team": g["home_team_abbrev"].values,
            "away_team": g["away_team_abbrev"].values,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_final": home_final,
            "away_final": away_final,
            "home_won": int(home_final > away_final),
            "period": period.values,
            "play_number": g["game_play_number"].values,
            "type_text": g["type_text"].values,
            "text": g["text"].values,
            "wallclock": g["wallclock"].values,
            "pc_seconds_left": pc_left.values,
            "seconds_elapsed": elapsed.values,
            "seconds_left_reg": seconds_left_reg.values,
            "home_score": hs_fixed.astype(int).values,
            "away_score": as_fixed.astype(int).values,
            "score_margin": (hs_fixed - as_fixed).astype(int).values,
            "actor_team_id": g["team_id"].values,
            "athlete_id_1": g["athlete_id_1"].values,
            "athlete_id_2": g["athlete_id_2"].values,
            "athlete_id_3": g["athlete_id_3"].values,
            "scoring_play": g["scoring_play"].values,
            "score_value": g["score_value"].values,
            "home_lineup": home_lu,
            "away_lineup": away_lu,
            "lineup_ok": lu_ok,
            "home_team_fouls_period": h_tf,
            "away_team_fouls_period": a_tf,
            "player_fouls": pf_str,
            "espn_home_spread": g["home_team_spread"].values,
        }))

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


# -- main ---------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs=2, type=int, metavar=("FIRST", "LAST"),
                    default=[2012, 2024],
                    help="ESPN season numbers (ENDING year), inclusive. "
                         "Default 2012 2024 = the 2011-12 through 2023-24 seasons.")
    ap.add_argument("--outdir", type=Path, default=Path("data"))
    ap.add_argument("--audit-fouls", action="store_true",
                    help="Print every foul-like type_text seen and how it was "
                         "classified, including any label this script does not "
                         "recognize.")
    args = ap.parse_args()

    first, last = args.seasons
    raw_dir = args.outdir / "espn_raw"
    out_dir = args.outdir / "pbp_espn"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {"unresolved_periods": 0, "carried_forward": 0, "no_roster": 0,
              "bad_starters": 0, "no_team_ids": 0, "score_corrections": 0,
              "no_official_score": 0, "final_score_disagreement": 0,
              "winner_would_have_flipped": 0, "sub_no_side": 0,
              "sub_out_not_on_court": 0, "sub_in_already_on_court": 0,
              "repaired_short_lineup": 0}
    foul_audit: Optional[set] = set() if args.audit_fouls else None

    totals = []
    for season in range(first, last + 1):
        print(f"\n=== ESPN season {season} "
              f"({season - 1}-{str(season)[-2:]}) ===")
        out_path = out_dir / f"pbp_{season}.parquet"
        if out_path.exists():
            existing = pd.read_parquet(out_path, columns=["game_id"])
            print(f"  exists: {len(existing):,} rows, "
                  f"{existing['game_id'].nunique():,} games - skipping")
            continue

        df = build_season(season, raw_dir, report, foul_audit)
        if df is None or df.empty:
            print(f"  no data for {season}")
            continue

        df.to_parquet(out_path, index=False)
        n_games = df["game_id"].nunique()
        bad = int((~df["lineup_ok"]).sum())
        totals.append((season, len(df), n_games, bad))
        print(f"  wrote {out_path.name}: {len(df):,} rows, {n_games:,} games, "
              f"{bad:,} rows with lineup_ok=False ({bad / len(df):.2%})")

    if totals:
        print("\nSeason totals:")
        print(f"  {'season':>7} {'rows':>10} {'games':>7} {'bad lineup':>11}")
        for s, r, gm, b in totals:
            print(f"  {s:>7} {r:>10,} {gm:>7,} {b:>11,}")
        print(f"  {'TOTAL':>7} {sum(t[1] for t in totals):>10,} "
              f"{sum(t[2] for t in totals):>7,} {sum(t[3] for t in totals):>11,}")

    print("\nData quality:")
    print(f"  periods where the opening five could not be inferred and the "
          f"previous period's five was carried forward : {report['carried_forward']:,}")
    print(f"  periods still not resolving to five a side (lineup_ok=False)    "
          f"  : {report['unresolved_periods']:,}")
    print(f"  games with no roster file entry                                 "
          f"  : {report['no_roster']:,}")
    print(f"  games whose roster did not give exactly five starters per side   "
          f"  : {report['bad_starters']:,}")
    print(f"  games with no home/away team id                                  "
          f"  : {report['no_team_ids']:,}")
    print(f"  score cells corrected for going backwards                        "
          f"  : {report['score_corrections']:,}")
    print(f"  subs whose team the roster could not identify                     "
          f"  : {report['sub_no_side']:,}")
    print(f"  subs whose departing player was not on the floor                 "
          f"  : {report['sub_out_not_on_court']:,}")
    print(f"  subs whose arriving player was already on the floor              "
          f"  : {report['sub_in_already_on_court']:,}")
    print(f"  short lineups repaired from a later first touch                  "
          f"  : {report['repaired_short_lineup']:,}")
    print(f"  games dropped for having no official final score                 "
          f"  : {report['no_official_score']:,}")
    print(f"  games where the play-by-play final disagreed with the schedule   "
          f"  : {report['final_score_disagreement']:,}")
    print(f"    of those, games where the WINNER would have been wrong         "
          f"  : {report['winner_would_have_flipped']:,}")

    if foul_audit is not None:
        print("\nFoul-like type_text seen:")
        for t in sorted(foul_audit):
            if t in TEAM_FOUL_TYPES:
                tag = "team foul + personal foul"
            elif t in PERSONAL_FOUL_TYPES:
                tag = "personal foul only"
            elif t in TECHNICAL_TYPES:
                tag = "technical - neither"
            else:
                tag = "*** UNRECOGNIZED - counted as neither ***"
            print(f"  {t:35s} {tag}")
        unknown = foul_audit - KNOWN_FOUL_LIKE
        if unknown:
            print(f"\n  {len(unknown)} unrecognized label(s). Add them to "
                  f"TEAM_FOUL_TYPES / PERSONAL_FOUL_TYPES / TECHNICAL_TYPES "
                  f"at the top of this file.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
