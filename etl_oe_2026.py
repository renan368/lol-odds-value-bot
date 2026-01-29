import pandas as pd

RAW_CSV = "./etl/database/2026_LoL_esports_match_data_from_OraclesElixir.csv"
TEAM_GAME_OUT = "./etl/database/team_game_agg_2026.csv"
MATCH_OUT = "./etl/database/match_agg_2026.csv"


def build_aggregates(raw_csv: str = RAW_CSV) -> tuple[pd.DataFrame, pd.DataFrame]:
    usecols = [
        "gameid", "league", "date", "side", "participantid",
        "teamname", "teamkills", "teamdeaths"
    ]

    df = pd.read_csv(raw_csv, usecols=usecols, low_memory=False)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["league"] = df["league"].astype("category")
    df["side"] = df["side"].astype("category")
    df["teamname"] = df["teamname"].astype("string")

    df["teamkills"] = pd.to_numeric(df["teamkills"], errors="coerce")
    df["teamdeaths"] = pd.to_numeric(df["teamdeaths"], errors="coerce")
    df["participantid"] = pd.to_numeric(df["participantid"], errors="coerce")

    # observed=True evita explosão de memória com categóricas (combinações não observadas). [web:84]
    team_game = (
        df.groupby(
            ["gameid", "league", "date", "side", "teamname"],
            as_index=False,
            sort=False,
            observed=True,
        )
        .agg(
            kills_for=("teamkills", "max"),
            kills_against=("teamdeaths", "max"),
            players=("participantid", "count"),
        )
    )

    blue = team_game[team_game["side"] == "Blue"][["gameid", "teamname", "kills_for", "kills_against"]].copy()
    red = team_game[team_game["side"] == "Red"][["gameid", "teamname", "kills_for", "kills_against"]].copy()

    blue = blue.rename(columns={
        "teamname": "blue_team",
        "kills_for": "blue_kills",
        "kills_against": "blue_deaths",
    })
    red = red.rename(columns={
        "teamname": "red_team",
        "kills_for": "red_kills",
        "kills_against": "red_deaths",
    })

    match = (
        team_game[["gameid", "league", "date"]].drop_duplicates()
        .merge(blue, on="gameid", how="left")
        .merge(red, on="gameid", how="left")
    )
    match["total_kills"] = match["blue_kills"] + match["red_kills"]

    return team_game, match


def save_aggregates(team_game: pd.DataFrame, match: pd.DataFrame) -> None:
    team_game.to_csv(TEAM_GAME_OUT, index=False)
    match.to_csv(MATCH_OUT, index=False)


def main():
    team_game, match = build_aggregates(RAW_CSV)

    # opcional: manter só times com 5 jogadores
    # team_game = team_game[team_game["players"] == 5]

    save_aggregates(team_game, match)
    print(f"OK: {TEAM_GAME_OUT}")
    print(f"OK: {MATCH_OUT}")


if __name__ == "__main__":
    main()
