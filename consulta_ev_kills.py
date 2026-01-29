import numpy as np
import pandas as pd

TEAM_GAME_OUT = "./etl/database/team_game_agg_2026.csv"
MATCH_OUT = "./etl/database/match_agg_2026.csv"


def normalize_team_name(x: str) -> str:
    return " ".join(str(x).strip().split()).lower()


def load_team_game(path: str = TEAM_GAME_OUT) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["teamname_norm"] = df["teamname"].map(normalize_team_name)
    return df


def team_profile(team_game: pd.DataFrame, team: str, league: str) -> dict:
    t = normalize_team_name(team)
    sub = team_game[(team_game["teamname_norm"] == t) & (team_game["league"] == league)].copy()

    if sub.empty:
        raise ValueError(f"Sem histórico para o time '{team}' na liga '{league}'.")

    return {
        "n_games": int(len(sub)),
        "kills_for_mean": float(sub["kills_for"].mean()),
        "kills_against_mean": float(sub["kills_against"].mean()),
        "kills_for_std": float(sub["kills_for"].std(ddof=1) if len(sub) > 1 else 3.0),
        "kills_against_std": float(sub["kills_against"].std(ddof=1) if len(sub) > 1 else 3.0),
    }


def expected_total_kills(profile_a: dict, profile_b: dict) -> float:
    exp_a = (profile_a["kills_for_mean"] + profile_b["kills_against_mean"]) / 2.0
    exp_b = (profile_b["kills_for_mean"] + profile_a["kills_against_mean"]) / 2.0
    return exp_a + exp_b


def league_total_kills_std(match_path: str = MATCH_OUT, league: str | None = None) -> float:
    df = pd.read_csv(match_path, low_memory=False)
    if league is not None:
        df = df[df["league"] == league]
    s = df["total_kills"].dropna()
    if len(s) <= 1:
        return 6.0
    return float(s.std(ddof=1))


def prob_over_under(
    exp_total: float,
    line: float,
    total_std: float,
    side: str,
    n_sims: int = 30000,
    seed: int = 42,
) -> float:
    rng = np.random.default_rng(seed)
    sims = rng.normal(loc=exp_total, scale=max(total_std, 1e-6), size=n_sims)

    if side.lower() == "over":
        return float(np.mean(sims > line))
    if side.lower() == "under":
        return float(np.mean(sims < line))
    raise ValueError("side deve ser 'over' ou 'under'.")


def fair_odds(p: float) -> float:
    if p <= 0 or p >= 1:
        raise ValueError("Probabilidade p deve estar entre 0 e 1.")
    return 1.0 / p


def ev_decimal(p: float, odds: float) -> float:
    # EV = p*odds - 1 (equivalente a EV = p*(odds-1) - (1-p)). [web:9]
    return p * odds - 1.0


def fair_line(exp_total: float) -> float:
    return float(np.floor(float(exp_total)) + 0.5)


def analyze_bet(team_game: pd.DataFrame, match: pd.DataFrame,
                team_a: str, team_b: str, league: str, line: float, side: str, odds_offered: float) -> dict:
    pa = team_profile(team_game, team_a, league)
    pb = team_profile(team_game, team_b, league)

    pa_blue = team_profile_side(team_game, team_a, league, side="Blue")
    pb_red  = team_profile_side(team_game, team_b, league, side="Red")

    a_blue = team_side_line_hits(match, team_a, league, side="Blue", line=line, bet_side=side)
    b_red  = team_side_line_hits(match, team_b, league, side="Red",  line=line, bet_side=side)

    a_general = team_general_line_hits(match, team_a, league, line=line, bet_side=side)
    b_general = team_general_line_hits(match, team_b, league, line=line, bet_side=side)

    exp_total = expected_total_kills(pa, pb)
    exp_total_side = expected_total_kills(pa_blue, pb_red)

    total_std = league_total_kills_std_df(match, league=league)

    p_side = prob_over_under(exp_total_side, line, total_std, side=side)
    odd_justa_side = fair_odds(p_side)
    ev_side = ev_decimal(p_side, odds_offered)


    p = prob_over_under(exp_total, line, total_std, side=side)
    odd_justa = fair_odds(p)
    ev = ev_decimal(p, odds_offered)

    h2h = h2h_stats(match, team_a, team_b, league, line, side)


    return {
        "league": league,
        "team_a": team_a,
        "team_b": team_b,
        "line_input": float(line),
        "side": side.lower(),
        "odds_offered": float(odds_offered),
        "p_model": p,
        "odd_justa_line_input": odd_justa,
        "ev": ev,
        "ev_pct": ev * 100.0,
        "expected_total_kills": exp_total,
        "linha_justa_model": fair_line(exp_total),
        "a_general_games_in_league": a_general["games"],
        "a_general_hits_line": a_general["hits"],
        "b_general_games_in_league": b_general["games"],
        "b_general_hits_line": b_general["hits"],
        "a_blue_games_in_league": a_blue["games"],
        "a_blue_hits_line": a_blue["hits"],
        "b_red_games_in_league": b_red["games"],
        "b_red_hits_line": b_red["hits"],
        "league_total_std": total_std,
        "h2h_games": h2h["h2h_games"],
        "h2h_hits": h2h["h2h_hits"],
        "h2h_hit_rate": (h2h["h2h_hits"] / h2h["h2h_games"] if h2h["h2h_games"] > 0 else None),
        # Modelo considerando side fixo (A=Blue, B=Red)
        "p_model_side": p_side,
        "odd_justa_line_input_side": odd_justa_side,
        "ev_side": ev_side,
        "ev_side_pct": ev_side * 100.0,
        "expected_total_kills_side": exp_total_side,
        "linha_justa_model_side": float(fair_line(exp_total_side)),
        "a_games_in_league_blue": pa_blue["n_games"],
        "b_games_in_league_red": pb_red["n_games"],
        "a_games_in_league": pa["n_games"],
        "b_games_in_league": pb["n_games"],


    }


def interactive_loop():
    print("Modo interativo (digite 'sair' para encerrar)")
    team_game = load_team_game()
    match = load_match()

    while True:
        league = input("Liga (ex: LIT): ").strip()
        if league.lower() == "sair":
            break

        team_a = input("Time A: ").strip()
        if team_a.lower() == "sair":
            break

        team_b = input("Time B: ").strip()
        if team_b.lower() == "sair":
            break

        side = input("Side (over/under): ").strip().lower()
        if side == "sair":
            break

        line_txt = input("Linha (ex: 25.5): ").strip()
        if line_txt.lower() == "sair":
            break

        odds_txt = input("Odd (decimal, ex: 1.95): ").strip()
        if odds_txt.lower() == "sair":
            break

        try:
            line = float(line_txt.replace(",", "."))
            odds = float(odds_txt.replace(",", "."))

            result = analyze_bet(team_game, match, team_a, team_b, league, line, side, odds)


            if result["a_games_in_league"] < 5 or result["b_games_in_league"] < 5:
                print("AVISO: amostra pequena para um dos times (menos de 5 jogos na liga).")

            print("\n--- Resultado ---")
            print(f"p_model: {result['p_model']:.4f}")
            print(f"odd_justa (linha enviada): {result['odd_justa_line_input']:.3f}")
            print(f"EV: {result['ev']:.4f} ({result['ev_pct']:.2f}%)")
            print(f"linha_justa_model: {result['linha_justa_model']:.1f}")
            print(f"expected_total_kills: {result['expected_total_kills']:.2f}")
            print(f"a_games_in_league: {result['a_games_in_league']} | b_games_in_league: {result['b_games_in_league']}")
            print(f"h2h: {result['h2h_hits']}/{result['h2h_games']} "
            f"({(result['h2h_hit_rate']*100):.1f}% se existir)" if result["h2h_games"] else "h2h: sem confrontos")
            print(f"A (Geral) linha: {result['a_general_games_in_league']}/{result['a_general_hits_line']} (games/hits)")
            print(f"B (Geral) linha: {result['b_general_games_in_league']}/{result['b_general_hits_line']} (games/hits)")

            print("\n--- Side fixo (A=Blue, B=Red) ---")
            print(f"p_model_side: {result['p_model_side']:.4f}")
            print(f"odd_justa_side (linha enviada): {result['odd_justa_line_input_side']:.3f}")
            print(f"EV_side: {result['ev_side']:.4f} ({result['ev_side_pct']:.2f}%)")
            print(f"linha_justa_model_side: {result['linha_justa_model_side']:.1f}")
            print(f"expected_total_kills_side: {result['expected_total_kills_side']:.2f}")
            print(f"a_games_in_league_blue: {result['a_games_in_league_blue']} | b_games_in_league_red: {result['b_games_in_league_red']}")
            print(f"A (Blue) linha: {result['a_blue_games_in_league']}/{result['a_blue_hits_line']} (games/hits)")
            print(f"B (Red)  linha: {result['b_red_games_in_league']}/{result['b_red_hits_line']} (games/hits)")

            print("-----------------\n")

        except Exception as e:
            print(f"\nErro: {e}\n")

def load_match(path: str = MATCH_OUT) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["blue_team_norm"] = df["blue_team"].map(normalize_team_name)
    df["red_team_norm"] = df["red_team"].map(normalize_team_name)
    return df

def h2h_stats(match: pd.DataFrame, team_a: str, team_b: str, league: str, line: float, side: str) -> dict:
    a = normalize_team_name(team_a)
    b = normalize_team_name(team_b)

    # jogos onde A vs B (qualquer lado), na liga
    h2h = match[
        (match["league"] == league) &
        (
            ((match["blue_team_norm"] == a) & (match["red_team_norm"] == b)) |
            ((match["blue_team_norm"] == b) & (match["red_team_norm"] == a))
        )
    ].copy()

    n = len(h2h)  # quantos confrontos [web:120]
    if n == 0:
        return {"h2h_games": 0, "h2h_hits": 0}

    if side.lower() == "over":
        hits = int((h2h["total_kills"] > line).sum())
    elif side.lower() == "under":
        hits = int((h2h["total_kills"] < line).sum())
    else:
        raise ValueError("side deve ser 'over' ou 'under'.")

    return {"h2h_games": n, "h2h_hits": hits}

def team_profile_side(team_game: pd.DataFrame, team: str, league: str, side: str) -> dict:
    t = normalize_team_name(team)

    sub = team_game[
        (team_game["teamname_norm"] == t) &
        (team_game["league"] == league) &
        (team_game["side"] == side)
    ].copy()

    if sub.empty:
        raise ValueError(f"Sem histórico para o time '{team}' na liga '{league}' jogando no side '{side}'.")

    return {
        "n_games": int(len(sub)),
        "kills_for_mean": float(sub["kills_for"].mean()),
        "kills_against_mean": float(sub["kills_against"].mean()),
        "kills_for_std": float(sub["kills_for"].std(ddof=1) if len(sub) > 1 else 3.0),
        "kills_against_std": float(sub["kills_against"].std(ddof=1) if len(sub) > 1 else 3.0),
    }

def team_side_line_hits(match: pd.DataFrame, team: str, league: str, side: str, line: float, bet_side: str) -> dict:
    t = normalize_team_name(team)

    if side == "Blue":
        sub = match[(match["league"] == league) & (match["blue_team_norm"] == t)].copy()
    elif side == "Red":
        sub = match[(match["league"] == league) & (match["red_team_norm"] == t)].copy()
    else:
        raise ValueError("side deve ser 'Blue' ou 'Red'.")

    total = len(sub)  # [web:120]
    if total == 0:
        return {"games": 0, "hits": 0}

    if bet_side.lower() == "over":
        hits = int((sub["total_kills"] > line).sum())
    elif bet_side.lower() == "under":
        hits = int((sub["total_kills"] < line).sum())
    else:
        raise ValueError("bet_side deve ser 'over' ou 'under'.")

    return {"games": total, "hits": hits}

def team_general_line_hits(match: pd.DataFrame, team: str, league: str, line: float, bet_side: str) -> dict:
    t = normalize_team_name(team)

    sub = match[
        (match["league"] == league) &
        ((match["blue_team_norm"] == t) | (match["red_team_norm"] == t))
    ].copy()

    total = len(sub)
    if total == 0:
        return {"games": 0, "hits": 0}

    if bet_side.lower() == "over":
        hits = int((sub["total_kills"] > line).sum())
    elif bet_side.lower() == "under":
        hits = int((sub["total_kills"] < line).sum())
    else:
        raise ValueError("bet_side deve ser 'over' ou 'under'.")

    return {"games": total, "hits": hits}


def league_total_kills_std_df(match: pd.DataFrame, league: str | None = None) -> float:
    df = match
    if league is not None:
        df = df[df["league"] == league]
    s = df["total_kills"].dropna()
    if len(s) <= 1:
        return 6.0
    return float(s.std(ddof=1))



def main():
    interactive_loop()


if __name__ == "__main__":
    main()
