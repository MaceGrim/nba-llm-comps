"""
Prep two JSON files for the blog page:

  viz_data.json   — per-model top-20 + agreement matrix (for chord + bump charts)
  pair_votes.json — compact pair → per-model-vote lookup (for inline prompt picker)

Reuses Elo / agreement logic from analyze.py. Run once after matchups.csv changes.
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
CSV_FILE = HERE / "matchups.csv"
PLAYERS_FILE = HERE / "players.json"
VIZ_OUT = HERE / "viz_data.json"
PAIRS_OUT = HERE / "pair_votes.json"

K_FACTOR = 32
INITIAL_ELO = 1500
MODEL_ORDER = ["gemma3:12b", "llama3.1:8b", "mistral:7b", "phi4:14b", "qwen2.5:7b"]


def expected_score(ra, rb):
    return 1.0 / (1.0 + math.pow(10, (rb - ra) / 400))


def update_elo(ra, rb, a_wins):
    ea = expected_score(ra, rb)
    sa = 1.0 if a_wins else 0.0
    return ra + K_FACTOR * (sa - ea), rb + K_FACTOR * ((1 - sa) - (1 - ea))


def main():
    with open(PLAYERS_FILE) as f:
        players = json.load(f)
    player_idx = {p: i for i, p in enumerate(players)}

    with open(CSV_FILE) as f:
        rows = [r for r in csv.DictReader(f) if r["result"] != "FAILED"]

    # --- per-model Elo ---
    model_elo = {m: {p: INITIAL_ELO for p in players} for m in MODEL_ORDER}
    for r in rows:
        m, pa, pb, pick = r["model"], r["player_a"], r["player_b"], r["result"]
        if m not in model_elo:
            continue
        a_wins = pick == pa
        model_elo[m][pa], model_elo[m][pb] = update_elo(model_elo[m][pa], model_elo[m][pb], a_wins)

    model_rankings = {}
    for m in MODEL_ORDER:
        ranked = sorted(model_elo[m].items(), key=lambda x: x[1], reverse=True)[:20]
        model_rankings[m] = [{"name": n, "rank": i + 1, "elo": round(e, 1)} for i, (n, e) in enumerate(ranked)]

    # --- agreement matrix ---
    matchup_votes = defaultdict(dict)  # matchup_id -> {model: pick}
    for r in rows:
        matchup_votes[r["matchup_id"]][r["model"]] = r["result"]

    agreement = {m1: {m2: None for m2 in MODEL_ORDER} for m1 in MODEL_ORDER}
    for m1 in MODEL_ORDER:
        for m2 in MODEL_ORDER:
            if m1 == m2:
                continue
            agree = total = 0
            for votes in matchup_votes.values():
                if m1 in votes and m2 in votes:
                    total += 1
                    if votes[m1] == votes[m2]:
                        agree += 1
            agreement[m1][m2] = round(agree / total * 100, 1) if total else 0.0

    # --- combined Elo across all judges (for player select ordering) ---
    combined = {p: INITIAL_ELO for p in players}
    for r in rows:
        pa, pb, pick = r["player_a"], r["player_b"], r["result"]
        a_wins = pick == pa
        combined[pa], combined[pb] = update_elo(combined[pa], combined[pb], a_wins)
    combined_ranking = [
        {"name": n, "rank": i + 1, "elo": round(e, 1)}
        for i, (n, e) in enumerate(sorted(combined.items(), key=lambda x: x[1], reverse=True))
    ]

    viz_data = {
        "models": MODEL_ORDER,
        "model_rankings": model_rankings,
        "agreement_matrix": agreement,
        "combined_ranking": combined_ranking,
    }
    VIZ_OUT.write_text(json.dumps(viz_data, indent=2))
    print(f"wrote {VIZ_OUT.name}: {len(model_rankings)} models × top-20, 5×5 agreement matrix")

    # --- pair votes: compact lookup ---
    # key: "min_idx,max_idx" (the order itertools.combinations emitted)
    # value: 5-char string, one char per model in MODEL_ORDER, A=player_a won, B=player_b won, ?=missing
    pair_votes = {}
    for r in rows:
        ai = player_idx.get(r["player_a"])
        bi = player_idx.get(r["player_b"])
        if ai is None or bi is None:
            continue
        # In matchups.csv, player_a / player_b match the order in the CSV row (the combinations order).
        lo, hi = (ai, bi) if ai < bi else (bi, ai)
        key = f"{lo},{hi}"
        if key not in pair_votes:
            pair_votes[key] = ["?"] * len(MODEL_ORDER)
        if r["model"] in MODEL_ORDER:
            m_i = MODEL_ORDER.index(r["model"])
            # 'A' = the lower-index player won, 'B' = the higher-index player won
            winner_idx = player_idx.get(r["result"])
            if winner_idx == lo:
                pair_votes[key][m_i] = "A"
            elif winner_idx == hi:
                pair_votes[key][m_i] = "B"
    pair_votes = {k: "".join(v) for k, v in pair_votes.items()}

    PAIRS_OUT.write_text(json.dumps({
        "models": MODEL_ORDER,
        "pairs": pair_votes,
    }))
    print(f"wrote {PAIRS_OUT.name}: {len(pair_votes)} pairs × {len(MODEL_ORDER)} models")


if __name__ == "__main__":
    main()
