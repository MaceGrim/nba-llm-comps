"""
Analyze matchups.csv — reconstruct Elo rankings from raw judge votes.

Builds:
  1. Combined Elo (every valid vote = one Elo update)
  2. Per-model Elo (each judge's votes build a separate ladder)
  3. Win rates by model
  4. Combined top-N vs per-model top-N comparison
  5. Agreement/disagreement matrix between judges
"""

import csv
import math
import json
from collections import defaultdict
from pathlib import Path

CSV_FILE = Path(__file__).parent / "matchups.csv"
PLAYERS_FILE = Path(__file__).parent / "players.json"
K_FACTOR = 32
INITIAL_ELO = 1500


def expected_score(ra, rb):
    return 1.0 / (1.0 + math.pow(10, (rb - ra) / 400))


def update_elo(ra, rb, a_wins):
    ea = expected_score(ra, rb)
    sa = 1.0 if a_wins else 0.0
    return ra + K_FACTOR * (sa - ea), rb + K_FACTOR * ((1 - sa) - (1 - ea))


def load_data():
    rows = []
    with open(CSV_FILE) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def print_ranking(title, elo_dict, top_n=50):
    ranked = sorted(elo_dict.items(), key=lambda x: x[1], reverse=True)
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")
    print(f"{'Rank':<6}{'Player':<30}{'Elo':>8}")
    print("-" * 60)
    for i, (name, elo) in enumerate(ranked[:top_n], 1):
        print(f"{i:<6}{name:<30}{elo:>8.1f}")
    return ranked


def analyze():
    rows = load_data()
    if not rows:
        print("No data in matchups.csv")
        return

    with open(PLAYERS_FILE) as f:
        all_players = json.load(f)

    judges = sorted(set(r["model"] for r in rows))
    valid_rows = [r for r in rows if r["result"] != "FAILED"]
    failed_rows = [r for r in rows if r["result"] == "FAILED"]
    # Support both "round" and "matchup_id" column names
    id_col = "matchup_id" if "matchup_id" in rows[0] else "round"
    matchups = set(r[id_col] for r in rows)

    print(f"Total rows: {len(rows)}")
    print(f"Valid votes: {len(valid_rows)}")
    print(f"Failed votes: {len(failed_rows)}")
    print(f"Matchups: {len(matchups)}")
    print(f"Judges: {', '.join(judges)}")

    # -------------------------------------------------------------------
    # 1a. Win-rate ranking (simple: wins / total appearances)
    # -------------------------------------------------------------------
    win_counts = defaultdict(int)
    appearance_counts = defaultdict(int)
    for r in valid_rows:
        pa, pb, pick = r["player_a"], r["player_b"], r["result"]
        appearance_counts[pa] += 1
        appearance_counts[pb] += 1
        win_counts[pick] += 1

    win_rates = {}
    for p in all_players:
        if appearance_counts[p] > 0:
            win_rates[p] = win_counts[p] / appearance_counts[p]
        else:
            win_rates[p] = 0.0

    wr_ranked = sorted(win_rates.items(), key=lambda x: x[1], reverse=True)
    print(f"\n{'=' * 60}")
    print("COMBINED RANKING (win rate)")
    print(f"{'=' * 60}")
    print(f"{'Rank':<6}{'Player':<30}{'Win%':>8}{'W':>6}{'L':>6}")
    print("-" * 60)
    for i, (name, wr) in enumerate(wr_ranked[:50], 1):
        w = win_counts[name]
        l = appearance_counts[name] - w
        print(f"{i:<6}{name:<30}{wr:>7.1%}{w:>6}{l:>6}")

    # -------------------------------------------------------------------
    # 1b. Combined Elo — every valid vote is an independent Elo update
    # -------------------------------------------------------------------
    combined_elo = {p: INITIAL_ELO for p in all_players}
    for r in valid_rows:
        pa, pb, pick = r["player_a"], r["player_b"], r["result"]
        a_wins = pick == pa
        combined_elo[pa], combined_elo[pb] = update_elo(
            combined_elo[pa], combined_elo[pb], a_wins
        )
    combined_ranked = print_ranking("COMBINED RANKING (Elo)", combined_elo)

    # -------------------------------------------------------------------
    # 2. Per-model Elo
    # -------------------------------------------------------------------
    model_elo = {j: {p: INITIAL_ELO for p in all_players} for j in judges}
    model_wins = {j: defaultdict(int) for j in judges}

    for r in valid_rows:
        j = r["model"]
        pa, pb, pick = r["player_a"], r["player_b"], r["result"]
        a_wins = pick == pa
        model_elo[j][pa], model_elo[j][pb] = update_elo(
            model_elo[j][pa], model_elo[j][pb], a_wins
        )
        model_wins[j][pick] += 1

    print(f"\n{'=' * 60}")
    print("PER-MODEL TOP 20")
    print(f"{'=' * 60}")
    model_rankings = {}
    for j in judges:
        ranked = sorted(model_elo[j].items(), key=lambda x: x[1], reverse=True)
        model_rankings[j] = ranked
        print(f"\n  {j}:")
        for i, (name, elo) in enumerate(ranked[:20], 1):
            print(f"    {i:>3}. {name:<28} {elo:>8.1f}")

    # -------------------------------------------------------------------
    # 3. Combined top-20 vs per-model top-20 overlap
    # -------------------------------------------------------------------
    combined_top20 = set(name for name, _ in combined_ranked[:20])

    print(f"\n{'=' * 60}")
    print("COMBINED TOP-20 vs PER-MODEL TOP-20 OVERLAP")
    print(f"{'=' * 60}")
    for j in judges:
        model_top20 = set(name for name, _ in model_rankings[j][:20])
        overlap = combined_top20 & model_top20
        unique = model_top20 - combined_top20
        print(f"\n  {j}: {len(overlap)}/20 overlap")
        if unique:
            print(f"    Only in this model's top-20: {', '.join(sorted(unique))}")

    # -------------------------------------------------------------------
    # 4. Win rates by model (total wins / total valid votes)
    # -------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("VOTE COUNTS BY MODEL")
    print(f"{'=' * 60}")
    for j in judges:
        total = sum(1 for r in rows if r["model"] == j)
        valid = sum(1 for r in valid_rows if r["model"] == j)
        fails = total - valid
        print(f"  {j:<30} {valid} valid, {fails} failed ({valid/total*100:.1f}% success)")

    # -------------------------------------------------------------------
    # 5. Agreement matrix — how often do pairs of judges agree?
    # -------------------------------------------------------------------
    # Group votes by matchup
    round_votes = defaultdict(dict)  # matchup_id -> {judge: pick}
    for r in valid_rows:
        round_votes[r[id_col]][r["model"]] = r["result"]

    print(f"\n{'=' * 60}")
    print("JUDGE AGREEMENT MATRIX (% same pick on shared matchups)")
    print(f"{'=' * 60}")
    # Header
    short_names = {j: j.split(":")[0][:12] for j in judges}
    header = f"{'':>20}" + "".join(f"{short_names[j]:>14}" for j in judges)
    print(header)
    for j1 in judges:
        row = f"{short_names[j1]:>20}"
        for j2 in judges:
            if j1 == j2:
                row += f"{'---':>14}"
                continue
            agree = 0
            total = 0
            for rnd, votes in round_votes.items():
                if j1 in votes and j2 in votes:
                    total += 1
                    if votes[j1] == votes[j2]:
                        agree += 1
            pct = (agree / total * 100) if total > 0 else 0
            row += f"{pct:>13.1f}%"
        print(row)

    # -------------------------------------------------------------------
    # 6. Most favored players per model
    # -------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("MOST FAVORED PLAYERS BY MODEL (top 10 by win count)")
    print(f"{'=' * 60}")
    for j in judges:
        top = sorted(model_wins[j].items(), key=lambda x: x[1], reverse=True)[:10]
        top_str = ", ".join(f"{name} ({w})" for name, w in top)
        print(f"  {j}:")
        print(f"    {top_str}")

    # -------------------------------------------------------------------
    # 7. Most controversial players (highest Elo variance across models)
    # -------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("MOST CONTROVERSIAL PLAYERS (highest Elo variance across models)")
    print(f"{'=' * 60}")
    player_variance = {}
    for p in all_players:
        elos = [model_elo[j][p] for j in judges]
        mean = sum(elos) / len(elos)
        var = sum((e - mean) ** 2 for e in elos) / len(elos)
        player_variance[p] = (var ** 0.5, elos, mean)

    controversial = sorted(player_variance.items(), key=lambda x: x[1][0], reverse=True)[:20]
    for name, (std, elos, mean) in controversial:
        elo_by_model = ", ".join(
            f"{j.split(':')[0][:8]}={model_elo[j][name]:.0f}" for j in judges
        )
        print(f"  {name:<28} mean={mean:.0f} std={std:.0f}  [{elo_by_model}]")


if __name__ == "__main__":
    analyze()
