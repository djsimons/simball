import pandas as pd
import json
import math
import os

# ── LOAD RAW DATA ─────────────────────────────────────────────────────────────
people   = pd.read_csv("People.csv")
batting  = pd.read_csv("Batting.csv")
pitching = pd.read_csv("Pitching.csv")
fielding = pd.read_csv("Fielding.csv")
war_bat  = pd.read_csv("war_bat.csv")
war_pit  = pd.read_csv("war_pit.csv")
teams    = pd.read_csv("Teams.csv")
awards_raw  = pd.read_csv("AwardsPlayers.csv")
allstar_raw = pd.read_csv("AllstarFull.csv")
hof_raw     = pd.read_csv("HallOfFame.csv")

# ── POSITION PLAYERS ──────────────────────────────────────────────────────────
pos_career = batting.groupby("playerID").agg(G=("G","sum")).reset_index()
pos_qualified = pos_career[pos_career["G"] >= 500].copy()
pos_qualified["role"] = "position"

# ── PITCHERS ──────────────────────────────────────────────────────────────────
pitch_career = pitching.groupby("playerID").agg(G=("G","sum"), GS=("GS","sum")).reset_index()
pitch_career["gs_rate"] = pitch_career["GS"] / pitch_career["G"]

starters  = pitch_career[(pitch_career["gs_rate"] >= 0.5) & (pitch_career["GS"] >= 100)].copy()
starters["role"] = "starter"
relievers = pitch_career[(pitch_career["gs_rate"] < 0.5) & (pitch_career["G"] >= 300)].copy()
relievers["role"] = "reliever"
pitchers_qualified = pd.concat([starters, relievers], ignore_index=True)

# ── DEDUPLICATE: pitchers take priority ───────────────────────────────────────
pitcher_ids = set(pitchers_qualified["playerID"])
pos_qualified = pos_qualified[~pos_qualified["playerID"].isin(pitcher_ids)]

pool = pd.concat([
    pos_qualified[["playerID","role"]],
    pitchers_qualified[["playerID","role"]]
], ignore_index=True)

# ── BIO DATA ──────────────────────────────────────────────────────────────────
bio = people[["playerID","nameFirst","nameLast","birthYear","birthCountry","birthCity","birthState","bats","throws"]].copy()
bio["fullName"] = (bio["nameFirst"].fillna("") + " " + bio["nameLast"].fillna("")).str.strip()

def handedness(row):
    b = str(row["bats"]).strip() if pd.notna(row["bats"]) else "?"
    t = str(row["throws"]).strip() if pd.notna(row["throws"]) else "?"
    if b == "nan": b = "?"
    if t == "nan": t = "?"
    return 'B' + b + '/T' + t

bio["handedness"] = bio.apply(handedness, axis=1)
pool = pool.merge(bio[["playerID","fullName","birthYear","birthCountry","birthCity","birthState","handedness"]], on="playerID", how="left")

# ── FIRST YEAR / LAST YEAR (for era filter) ───────────────────────────────────
bat_years = batting.groupby("playerID")["yearID"].agg(["min","max"]).reset_index()
bat_years.columns = ["playerID","firstYear_bat","lastYear_bat"]

pit_years = pitching.groupby("playerID")["yearID"].agg(["min","max"]).reset_index()
pit_years.columns = ["playerID","firstYear_pit","lastYear_pit"]

pool = pool.merge(bat_years, on="playerID", how="left")
pool = pool.merge(pit_years, on="playerID", how="left")

def get_first_year(row):
    years = [y for y in [row.get("firstYear_bat"), row.get("firstYear_pit")] if pd.notna(y)]
    return int(min(years)) if years else None

def get_last_year(row):
    years = [y for y in [row.get("lastYear_bat"), row.get("lastYear_pit")] if pd.notna(y)]
    return int(max(years)) if years else None

pool["firstYear"] = pool.apply(get_first_year, axis=1)
pool["lastYear"]  = pool.apply(get_last_year,  axis=1)
pool = pool.drop(columns=["firstYear_bat","lastYear_bat","firstYear_pit","lastYear_pit"])

# ── MLB ID for photos ─────────────────────────────────────────────────────────
mlb_ids = war_bat[["player_ID","mlb_ID"]].dropna(subset=["mlb_ID"]).drop_duplicates("player_ID")
mlb_ids.columns = ["playerID","mlbID"]
mlb_ids["mlbID"] = mlb_ids["mlbID"].astype(int)
pool = pool.merge(mlb_ids, on="playerID", how="left")

# Also check war_pit for IDs missing from war_bat
mlb_ids_pit = war_pit[["player_ID","mlb_ID"]].dropna(subset=["mlb_ID"]).drop_duplicates("player_ID")
mlb_ids_pit.columns = ["playerID","mlbID_pit"]
mlb_ids_pit["mlbID_pit"] = mlb_ids_pit["mlbID_pit"].astype(int)
pool = pool.merge(mlb_ids_pit, on="playerID", how="left")
pool["mlbID"] = pool["mlbID"].combine_first(pool["mlbID_pit"])
pool = pool.drop(columns=["mlbID_pit"])

# ── NAME+BIRTHYEAR FALLBACK for mismatched IDs (e.g. sabatcc01 vs sabatc.01) ─
# Build a lookup from war files: (normalized_name, birthYear) -> (bref_id, mlb_ID)
def norm_name(n):
    if not isinstance(n, str): return ""
    return n.lower().strip()

# Get birthYear from people for war file players
war_all = pd.concat([
    war_bat[["player_ID","name_common","mlb_ID"]].drop_duplicates("player_ID"),
    war_pit[["player_ID","name_common","mlb_ID"]].drop_duplicates("player_ID"),
]).drop_duplicates("player_ID")

# Merge with people to get birthYear by bref player_ID
people_by_id = people[["playerID","birthYear"]].copy()
people_by_id.columns = ["player_ID","birthYear_war"]
war_all = war_all.merge(people_by_id, on="player_ID", how="left")

# Also try joining by name+birthYear from people
people_norm = people[["playerID","nameFirst","nameLast","birthYear"]].copy()
people_norm["fullName"] = (people_norm["nameFirst"].fillna("") + " " + people_norm["nameLast"].fillna("")).str.strip()
people_norm["name_norm"] = people_norm["fullName"].apply(norm_name)

war_all["name_norm"] = war_all["name_common"].apply(norm_name)

# For pool players missing mlbID, try name+birthYear match against war_all
missing_mlb = pool[pool["mlbID"].isna()].copy()
if len(missing_mlb):
    # Add normalized name and birthYear to pool
    pool_norm = pool[["playerID","fullName","birthYear"]].copy()
    pool_norm["name_norm"] = pool_norm["fullName"].apply(norm_name)

    # Build fallback lookup from war_all
    war_lookup = war_all[war_all["mlb_ID"].notna()].copy()
    war_lookup["mlb_ID"] = war_lookup["mlb_ID"].astype(int)

    # Join missing players by name_norm + birthYear
    fallback = missing_mlb[["playerID","fullName","birthYear"]].copy()
    fallback["name_norm"] = fallback["fullName"].apply(norm_name)
    fallback = fallback.merge(
        war_lookup[["name_norm","mlb_ID","player_ID"]].rename(columns={"player_ID":"bref_id_fallback"}),
        on="name_norm", how="left"
    )
    fallback = fallback.drop_duplicates("playerID")
    fallback = fallback[["playerID","mlb_ID","bref_id_fallback"]].rename(columns={"mlb_ID":"mlbID_fallback"})

    pool = pool.merge(fallback, on="playerID", how="left")
    pool["mlbID"] = pool["mlbID"].combine_first(pool["mlbID_fallback"])
    pool = pool.drop(columns=["mlbID_fallback","bref_id_fallback"], errors="ignore")

    recovered = pool["mlbID"].notna().sum() - (len(pool) - missing_mlb["mlbID"].isna().sum())
    print("mlbID fallback recovered: {}".format(recovered))

# ── WAR: assign by role from pool, not from pitcher_ids set ──────────────────
# Build WAR lookup by playerID first, then fall back to name+birthYear
bat_war = war_bat.groupby("player_ID")["WAR"].sum().reset_index()
bat_war.columns = ["playerID","WAR_bat"]

pit_war = war_pit.groupby("player_ID")["WAR"].sum().reset_index()
pit_war.columns = ["playerID","WAR_pit"]

# Use outer join but do NOT fillna -- keep NaN distinct from 0
all_war = bat_war.merge(pit_war, on="playerID", how="outer")
pool = pool.merge(all_war, on="playerID", how="left")

# Fallback: for players still missing WAR, try matching by name_norm
missing_war = pool[pool["WAR_bat"].isna() & pool["WAR_pit"].isna()].copy()
if len(missing_war):
    pool_norm2 = missing_war[["playerID","fullName"]].copy()
    pool_norm2["name_norm"] = pool_norm2["fullName"].apply(norm_name)

    bat_war_named = war_bat.groupby("player_ID").agg(
        WAR_bat=("WAR","sum"), name_common=("name_common","first")
    ).reset_index()
    bat_war_named["name_norm"] = bat_war_named["name_common"].apply(norm_name)

    pit_war_named = war_pit.groupby("player_ID").agg(
        WAR_pit=("WAR","sum"), name_common=("name_common","first")
    ).reset_index()
    pit_war_named["name_norm"] = pit_war_named["name_common"].apply(norm_name)

    # Merge bat fallback
    fb_bat = pool_norm2.merge(bat_war_named[["name_norm","WAR_bat"]], on="name_norm", how="left").drop_duplicates("playerID")
    fb_pit = pool_norm2.merge(pit_war_named[["name_norm","WAR_pit"]], on="name_norm", how="left").drop_duplicates("playerID")

    fb = fb_bat[["playerID","WAR_bat"]].merge(fb_pit[["playerID","WAR_pit"]], on="playerID", how="outer")
    fb.columns = ["playerID","WAR_bat_fb","WAR_pit_fb"]

    pool = pool.merge(fb, on="playerID", how="left")
    pool["WAR_bat"] = pool["WAR_bat"].combine_first(pool["WAR_bat_fb"])
    pool["WAR_pit"] = pool["WAR_pit"].combine_first(pool["WAR_pit_fb"])
    pool = pool.drop(columns=["WAR_bat_fb","WAR_pit_fb"], errors="ignore")
    print("WAR fallback applied to {} players".format(len(missing_war)))

# Fallback: stripped playerID match (handles apostrophes/periods in IDs)
# e.g. oneilpa01 vs o'neipa01, sabatcc01 vs sabatc.01
missing_war2 = pool[pool["WAR_bat"].isna() & pool["WAR_pit"].isna()].copy()
if len(missing_war2):
    def strip_id(pid):
        import re
        return re.sub(r'[^a-z0-9]', '', str(pid).lower())

    # Build stripped ID -> WAR lookup from both war files
    bat_war_strip = war_bat.groupby("player_ID")["WAR"].sum().reset_index()
    bat_war_strip.columns = ["player_ID", "WAR_bat"]
    bat_war_strip["id_stripped"] = bat_war_strip["player_ID"].apply(strip_id)

    pit_war_strip = war_pit.groupby("player_ID")["WAR"].sum().reset_index()
    pit_war_strip.columns = ["player_ID", "WAR_pit"]
    pit_war_strip["id_stripped"] = pit_war_strip["player_ID"].apply(strip_id)

    # Add stripped ID to missing players
    missing_war2["id_stripped"] = missing_war2["playerID"].apply(strip_id)

    fb2_bat = missing_war2[["playerID","id_stripped"]].merge(
        bat_war_strip[["id_stripped","WAR_bat"]], on="id_stripped", how="left"
    ).drop_duplicates("playerID")
    fb2_pit = missing_war2[["playerID","id_stripped"]].merge(
        pit_war_strip[["id_stripped","WAR_pit"]], on="id_stripped", how="left"
    ).drop_duplicates("playerID")

    fb2 = fb2_bat[["playerID","WAR_bat"]].merge(
        fb2_pit[["playerID","WAR_pit"]], on="playerID", how="outer"
    )
    fb2.columns = ["playerID","WAR_bat_fb2","WAR_pit_fb2"]

    pool = pool.merge(fb2, on="playerID", how="left")
    pool["WAR_bat"] = pool["WAR_bat"].combine_first(pool["WAR_bat_fb2"])
    pool["WAR_pit"] = pool["WAR_pit"].combine_first(pool["WAR_pit_fb2"])
    pool = pool.drop(columns=["WAR_bat_fb2","WAR_pit_fb2"], errors="ignore")

    recovered2 = pool[pool["WAR_bat"].notna() | pool["WAR_pit"].notna()]["playerID"].isin(missing_war2["playerID"]).sum()
    print("WAR stripped-ID fallback recovered: {}".format(recovered2))

# Fallback: truncated ID match (handles BBref's period-truncation convention)
# e.g. Lahman burneaj01 -> trunc burnea01, BBref burnea.01 -> trunc burnea01
missing_war3 = pool[pool["WAR_bat"].isna() & pool["WAR_pit"].isna()].copy()
if len(missing_war3):
    import re

    def trunc_id(pid):
        pid = str(pid).lower()
        clean = re.sub(r'[^a-z0-9]', '', pid)
        alpha = re.sub(r'\d', '', clean)
        digits = re.sub(r'[a-z]', '', clean)
        return alpha[:6] + digits[-2:]

    bat_war_trunc = war_bat.groupby("player_ID")["WAR"].sum().reset_index()
    bat_war_trunc.columns = ["player_ID", "WAR_bat"]
    bat_war_trunc["id_trunc"] = bat_war_trunc["player_ID"].apply(trunc_id)

    pit_war_trunc = war_pit.groupby("player_ID")["WAR"].sum().reset_index()
    pit_war_trunc.columns = ["player_ID", "WAR_pit"]
    pit_war_trunc["id_trunc"] = pit_war_trunc["player_ID"].apply(trunc_id)

    missing_war3["id_trunc"] = missing_war3["playerID"].apply(trunc_id)

    fb3_bat = missing_war3[["playerID","id_trunc"]].merge(
        bat_war_trunc[["id_trunc","WAR_bat"]], on="id_trunc", how="left"
    ).drop_duplicates("playerID")
    fb3_pit = missing_war3[["playerID","id_trunc"]].merge(
        pit_war_trunc[["id_trunc","WAR_pit"]], on="id_trunc", how="left"
    ).drop_duplicates("playerID")

    fb3 = fb3_bat[["playerID","WAR_bat"]].merge(
        fb3_pit[["playerID","WAR_pit"]], on="playerID", how="outer"
    )
    fb3.columns = ["playerID","WAR_bat_fb3","WAR_pit_fb3"]

    pool = pool.merge(fb3, on="playerID", how="left")
    pool["WAR_bat"] = pool["WAR_bat"].combine_first(pool["WAR_bat_fb3"])
    pool["WAR_pit"] = pool["WAR_pit"].combine_first(pool["WAR_pit_fb3"])
    pool = pool.drop(columns=["WAR_bat_fb3","WAR_pit_fb3"], errors="ignore")

    recovered3 = pool[pool["WAR_bat"].notna() | pool["WAR_pit"].notna()]["playerID"].isin(missing_war3["playerID"]).sum()
    print("WAR truncated-ID fallback recovered: {}".format(recovered3))

def pick_war(row):
    # Use role from pool directly -- avoids pitcher_ids set scope issues
    if row["role"] in ("starter","reliever"):
        val = row["WAR_pit"]
        return round(float(val), 1) if pd.notna(val) else None
    else:
        val = row["WAR_bat"]
        return round(float(val), 1) if pd.notna(val) else None

pool["WAR"] = pool.apply(pick_war, axis=1)
pool = pool.drop(columns=["WAR_bat","WAR_pit"])

# ── MANUAL WAR OVERRIDES ──────────────────────────────────────────────────────
# Players whose Lahman/BBref IDs don't reconcile via any fallback
MANUAL_WAR = {
    "drewjd01":  44.9,  # J.D. Drew
    "snowjt01":  11.0,  # J.T. Snow
    "ryanbj01":  11.6,  # B.J. Ryan
    "garcifr02": 34.2,  # Freddy Garcia (Mariners, b.1976)
    "harriwi02":  7.7,  # Will Harris (Astros reliever, b.1984)
    "hensleg01": 16.8,  # Eggie/Logan Hensley (BBref: hensllo01)
    "graydo02":  23.2,  # Sam Gray (Browns, b.1897; BBref: graysa01)
}
for pid, war in MANUAL_WAR.items():
    mask = pool["playerID"] == pid
    if mask.any():
        pool.loc[mask, "WAR"] = war
        print("Manual WAR override: {} -> {}".format(pid, war))
    else:
        print("WARNING: {} not found in pool for manual WAR override".format(pid))

# ── PRIMARY POSITION + POSITIONS PLAYED ──────────────────────────────────────
field = fielding.copy()
field["POS"] = field["POS"].str.strip()

# Clean OF sub-positions everywhere
def clean_pos(p):
    if p in ("LF","CF","RF"): return "OF"
    return p

field["POS"] = field["POS"].apply(clean_pos)

pos_totals = field.groupby(["playerID","POS"]).agg(
    G_pos=("G","sum"), Inn=("InnOuts","sum")
).reset_index()

# Primary position
pos_sorted = pos_totals.sort_values(["playerID","G_pos","Inn"], ascending=[True,False,False])
primary_pos = pos_sorted.groupby("playerID").first().reset_index()[["playerID","POS"]]
primary_pos.columns = ["playerID","primaryPos"]
pool = pool.merge(primary_pos, on="playerID", how="left")

# Positions played with 10+ games (already cleaned to OF)
pos_10g = pos_totals[pos_totals["G_pos"] >= 10].groupby("playerID")["POS"].apply(list).reset_index()
pos_10g.columns = ["playerID","positionsPlayed"]
pool = pool.merge(pos_10g, on="playerID", how="left")

# Override pitchers
pool.loc[pool["role"] == "starter",  "primaryPos"] = "SP"
pool.loc[pool["role"] == "reliever", "primaryPos"] = "RP"
pool.loc[pool["role"].isin(["starter","reliever"]), "positionsPlayed"] = \
    pool.loc[pool["role"].isin(["starter","reliever"]), "role"].apply(
        lambda r: ["SP"] if r == "starter" else ["RP"]
    )

# ── PRIMARY TEAM NAME (year-aware) ────────────────────────────────────────────
team_name_lookup = teams.set_index(["teamID","yearID"])["name"].to_dict()

def get_primary_team(df):
    tg  = df.groupby(["playerID","teamID","yearID"])["G"].sum().reset_index()
    tot = tg.groupby(["playerID","teamID"])["G"].sum().reset_index()
    tot = tot.sort_values(["playerID","G"], ascending=[True,False])
    best = tot.groupby("playerID").first().reset_index()[["playerID","teamID"]]
    merged = best.merge(tg[["playerID","teamID","yearID"]], on=["playerID","teamID"], how="left")
    last_year = merged.groupby(["playerID","teamID"])["yearID"].max().reset_index()
    best = best.merge(last_year, on=["playerID","teamID"], how="left")
    def lookup(row):
        return team_name_lookup.get((row["teamID"], row["yearID"]), row["teamID"])
    best["primaryTeam"] = best.apply(lookup, axis=1)
    return best[["playerID","primaryTeam"]]

bat_teams = get_primary_team(batting)
pit_teams = get_primary_team(pitching)
all_primary = pd.concat([bat_teams, pit_teams]).drop_duplicates("playerID")
pool = pool.merge(all_primary, on="playerID", how="left")

# ── NEGRO LEAGUE FLAG ─────────────────────────────────────────────────────────
NL_TEAM_IDS = {
    'CL5','CCG','BRG','CAG','NLG','SLG','ABC','CSE','CSW','AC','HIL','DS',
    'KCM','COB','SLS','ACB','BBB','MRS','HBG','HG','AB2','CC2','NBY','PC',
    'PS','NYC','NE','DTS','SL2','AB3','BEG','JRC','SL3','CBE','CC',
}

bat_nl = set(batting[batting["teamID"].isin(NL_TEAM_IDS)]["playerID"])
pit_nl = set(pitching[pitching["teamID"].isin(NL_TEAM_IDS)]["playerID"])
nl_players = bat_nl | pit_nl
pool["isNegroLeague"] = pool["playerID"].isin(nl_players)
print("Negro League players in pool:", pool["isNegroLeague"].sum())

# ── CAREER BATTING STATS ──────────────────────────────────────────────────────
bat_totals = batting.groupby("playerID").agg(
    G_bat=("G","sum"), AB=("AB","sum"), R=("R","sum"), H=("H","sum"),
    D=("2B","sum"), T=("3B","sum"), HR=("HR","sum"), RBI=("RBI","sum"),
    BB=("BB","sum"), SO=("SO","sum"), SB=("SB","sum"),
    HBP=("HBP","sum"), SF=("SF","sum"), SH=("SH","sum"),
).reset_index()
bat_totals["PA"]  = (bat_totals["AB"] + bat_totals["BB"] +
                     bat_totals["HBP"].fillna(0) +
                     bat_totals["SF"].fillna(0) +
                     bat_totals["SH"].fillna(0)).round(0).astype("Int64")
bat_totals["BA"]  = (bat_totals["H"] / bat_totals["AB"]).round(3)
bat_totals["SLG"] = ((bat_totals["H"] + bat_totals["D"] +
                       2*bat_totals["T"] + 3*bat_totals["HR"]) /
                      bat_totals["AB"]).round(3)
pool = pool.merge(bat_totals, on="playerID", how="left")

# ── CAREER PITCHING STATS ─────────────────────────────────────────────────────
pit_totals = pitching.groupby("playerID").agg(
    W=("W","sum"), L=("L","sum"), G_pit=("G","sum"), GS_pit=("GS","sum"),
    GR=("G","sum"),  # will compute GR = G - GS below
    SV=("SV","sum"), IPouts=("IPouts","sum"), H_pit=("H","sum"),
    BB_pit=("BB","sum"), SO_pit=("SO","sum"), ER=("ER","sum"),
).reset_index()
pit_totals["IP"]  = (pit_totals["IPouts"] / 3).round(1)
pit_totals["ERA"] = ((pit_totals["ER"] * 9) /
                      pit_totals["IP"].replace(0, float("nan"))).round(2)
pit_totals["GR"]  = pit_totals["G_pit"] - pit_totals["GS_pit"]
pool = pool.merge(pit_totals, on="playerID", how="left")

# ── SANITY CHECK ──────────────────────────────────────────────────────────────
print("Pool size:", len(pool))
print(pool["role"].value_counts())
print("Missing WAR:", pool["WAR"].isna().sum())
print("Missing firstYear:", pool["firstYear"].isna().sum())

# Check Nagy specifically
nagy = pool[pool["playerID"] == "nagych01"]
if len(nagy):
    print("\nNagy WAR:", nagy["WAR"].values[0], "role:", nagy["role"].values[0])

print("\nSample:")
print(pool[["fullName","role","primaryPos","primaryTeam","WAR","firstYear","lastYear"]].sample(10).to_string(index=False))

# ── AWARDS ────────────────────────────────────────────────────────────────────
# All-Star selections: count unique years per player
allstar_counts = allstar_raw.groupby("playerID")["yearID"].nunique().reset_index()
allstar_counts.columns = ["playerID", "allStars"]

# MVP, Cy Young, Rookie of the Year: count wins per player
mvp = awards_raw[awards_raw["awardID"] == "Most Valuable Player"].groupby("playerID").size().reset_index(name="mvp")
cy  = awards_raw[awards_raw["awardID"] == "Cy Young Award"].groupby("playerID").size().reset_index(name="cy")
roy = awards_raw[awards_raw["awardID"] == "Rookie of the Year"].groupby("playerID").size().reset_index(name="roy")

# Hall of Fame: players only, inducted
hof = hof_raw[(hof_raw["inducted"] == "Y") & (hof_raw["category"] == "Player")][["playerID"]].copy()
hof["hof"] = True

# Merge all into pool
pool = pool.merge(allstar_counts, on="playerID", how="left")
pool = pool.merge(mvp, on="playerID", how="left")
pool = pool.merge(cy,  on="playerID", how="left")
pool = pool.merge(roy, on="playerID", how="left")
pool = pool.merge(hof, on="playerID", how="left")

pool["allStars"] = pool["allStars"].fillna(0).astype(int)
pool["mvp"]      = pool["mvp"].fillna(0).astype(int)
pool["cy"]       = pool["cy"].fillna(0).astype(int)
pool["roy"]      = pool["roy"].fillna(0).astype(int)
pool["hof"]      = pool["hof"].fillna(False).astype(bool)

# Sanity check
hof_count = pool["hof"].sum()
as_count  = (pool["allStars"] > 0).sum()
print(f"HOF players in pool: {hof_count}")
print(f"All-Star players in pool: {as_count}")

# ── 3-LETTER COUNTRY CODES ────────────────────────────────────────────────────
COUNTRY_CODES = {
    "USA": "USA", "D.R.": "DOM", "Venezuela": "VEN", "Cuba": "CUB",
    "Panama": "PAN", "Puerto Rico": "PUR", "P.R.": "PUR", "P.R": "PUR",
    "Mexico": "MEX", "MÉX": "MEX", "MEX": "MEX",
    "Canada": "CAN", "Japan": "JPN", "South Korea": "KOR", "Korea": "KOR",
    "Australia": "AUS", "Colombia": "COL", "Nicaragua": "NIC", "Aruba": "ARU",
    "Netherlands": "NED", "Curacao": "CUR", "Brazil": "BRA", "Germany": "GER",
    "Spain": "ESP", "Italy": "ITA", "Honduras": "HON", "Bahamas": "BAH",
    "Jamaica": "JAM", "Taiwan": "TPE", "Virgin Islands": "VIR",
    "Dominican Republic": "DOM", "Haiti": "HAI", "England": "ENG",
    "Scotland": "SCO", "Ireland": "IRL", "France": "FRA", "Sweden": "SWE",
    "Russia": "RUS", "Poland": "POL", "China": "CHN", "Philippines": "PHL",
    "Belize": "BLZ", "Costa Rica": "CRC", "Guatemala": "GUA", "Peru": "PER",
    "Ecuador": "ECU", "Bolivia": "BOL", "Argentina": "ARG", "Chile": "CHI",
    "Uruguay": "URU", "Trinidad": "TRI", "Barbados": "BAR",
    # Lahman legacy values
    "West Germany": "GER", "Czechoslovakia": "CZE", "Yugoslavia": "YUG",
    "UK": "GBR",
}

def country_code(c):
    if not c or (isinstance(c, float) and math.isnan(c)):
        return None
    return COUNTRY_CODES.get(c, c[:3].upper())

# ── EXPORT JSON ───────────────────────────────────────────────────────────────
def v(x):
    if isinstance(x, float) and math.isnan(x):
        return None
    if x is None:
        return None
    try:
        if float(x) == int(float(x)):
            return int(float(x))
    except:
        pass
    return x

players = []
for _, row in pool.iterrows():
    pp = row["positionsPlayed"]
    if not isinstance(pp, list):
        pp = []

    p = {
        "id":              row["playerID"],
        "name":            row["fullName"],
        "birthYear":       v(row["birthYear"]),
        "birthCountry":    country_code(row["birthCountry"]),
        "birthCity":       row["birthCity"] if pd.notna(row.get("birthCity")) else None,
        "birthState":      row["birthState"] if pd.notna(row.get("birthState")) else None,
        "handedness":      row["handedness"] if pd.notna(row["handedness"]) else None,
        "role":            row["role"],
        "primaryPos":      row["primaryPos"]  if pd.notna(row["primaryPos"])  else None,
        "positionsPlayed": pp,
        "primaryTeam":     row["primaryTeam"] if pd.notna(row["primaryTeam"]) else None,
        "isNegroLeague":   bool(row["isNegroLeague"]),
        "mlbID":           v(row["mlbID"]),
        "WAR":             None if (row["WAR"] is None or (isinstance(row["WAR"], float) and math.isnan(row["WAR"]))) else round(float(row["WAR"]), 1),
        "firstYear":       v(row["firstYear"]),
        "lastYear":        v(row["lastYear"]),
        "G":               v(row["G_bat"]),
        "PA":              v(row["PA"]),
        "AB":              v(row["AB"]),
        "R":               v(row["R"]),
        "H":               v(row["H"]),
        "D":               v(row["D"]),
        "T":               v(row["T"]),
        "HR":              v(row["HR"]),
        "RBI":             v(row["RBI"]),
        "BB":              v(row["BB"]),
        "SO":              v(row["SO"]),
        "SB":              v(row["SB"]),
        "BA":              v(row["BA"]),
        "SLG":             v(row["SLG"]),
        "W":               v(row["W"]),
        "L":               v(row["L"]),
        "SV":              v(row["SV"]),
        "IP":              v(row["IP"]),
        "GS_pit":          v(row["GS_pit"]),
        "GR":              v(row["GR"]),
        "SO_pit":          v(row["SO_pit"]),
        "BB_pit":          v(row["BB_pit"]),
        "ERA":             v(row["ERA"]),
        "allStars":        int(row["allStars"]),
        "mvp":             int(row["mvp"]),
        "cy":              int(row["cy"]),
        "roy":             int(row["roy"]),
        "hof":             bool(row["hof"]),
    }
    players.append(p)

# ── TWO-WAY SPECIAL CASES: Ohtani and Ruth ───────────────────────────────────
# Each gets two entries: one as their primary role, one as the other side.
# We find their existing entry and clone it with swapped role/stats.

TWO_WAY = [
    ("ohtansh01", "Shohei Ohtani"),
    ("ruthba01",  "Babe Ruth"),
]

existing = {p["id"]: p for p in players}

for pid, name in TWO_WAY:
    orig = existing.get(pid)
    if not orig:
        print("WARNING: {} not found in pool".format(name))
        continue

    clone = dict(orig)
    clone["id"] = pid + "_alt"
    clone["name"] = name  # same name, dropdown will show pos/year to distinguish

    if orig["role"] == "position":
        # Add pitcher version
        clone["role"] = "starter"
        clone["primaryPos"] = "SP"
        clone["positionsPlayed"] = ["SP"]
        # Keep pitching stats, null out batting stats for similarity calc
        clone["G"] = None; clone["AB"] = None; clone["R"] = None
        clone["H"] = None; clone["D"] = None; clone["T"] = None
        clone["HR"] = None; clone["RBI"] = None; clone["BB"] = None
        clone["SO"] = None; clone["SB"] = None; clone["BA"] = None; clone["SLG"] = None
        # Use pitching WAR from war_pit
        pit_row = war_pit[war_pit["player_ID"] == pid]
        clone["WAR"] = round(float(pit_row["WAR"].sum()), 1) if len(pit_row) else None
    else:
        # Add position version
        clone["role"] = "position"
        clone["primaryPos"] = "OF"  # both Ruth and Ohtani are OF as hitters
        clone["positionsPlayed"] = ["OF","1B"]
        # Keep batting stats, null out pitching stats
        clone["W"] = None; clone["L"] = None; clone["SV"] = None
        clone["IP"] = None; clone["GS_pit"] = None; clone["GR"] = None
        clone["SO_pit"] = None; clone["ERA"] = None
        # Use batting WAR
        bat_row = war_bat[war_bat["player_ID"] == pid]
        clone["WAR"] = round(float(bat_row["WAR"].sum()), 1) if len(bat_row) else None

    players.append(clone)
    print("Added alt entry: {} ({})".format(clone["name"], clone["role"]))

with open("players.json", "w") as f:
    json.dump(players, f, indent=2)

print("\nExported {} players to players.json".format(len(players)))
print("File size: {} MB".format(round(os.path.getsize("players.json") / 1024 / 1024, 2)))
