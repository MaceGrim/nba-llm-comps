"""
NBA GOAT Ranker — Round-robin pairwise comparisons via LLM judges.

Every pair of players is evaluated by every model. Raw votes are saved
to matchups.csv (one row per model per pair). Elo and all other metrics
are reconstructed from this CSV by analyze.py.

Models are run one at a time (Mac swaps models in/out of memory).
Within each model, matchups run in parallel since the model is loaded.
"""

import csv
import itertools
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
API_URL = f"{OLLAMA_BASE}/v1/chat/completions"

JUDGES = [
    {"model": "gemma3:12b", "name": "gemma3:12b"},
    {"model": "llama3.1:8b", "name": "llama3.1:8b"},
    {"model": "phi4:14b", "name": "phi4:14b"},
    {"model": "mistral:7b", "name": "mistral:7b"},
    {"model": "qwen2.5:7b", "name": "qwen2.5:7b"},
]

SYSTEM_MESSAGE = {
    "role": "system",
    "content": "You are a sports analyst. When asked to compare players, you must pick one. No hedging.",
}

PARALLEL_PER_MODEL = 8
TEMPERATURE = 0.3
REQUEST_TIMEOUT = 300

PLAYERS_FILE = Path(__file__).parent / "players.json"
PROGRESS_FILE = Path(__file__).parent / "progress.json"
CSV_FILE = Path(__file__).parent / "matchups.csv"

PROMPT_TEMPLATE = (
    "Who is the better NBA player: {player_a} or {player_b}?\n"
    "Consider career achievements, impact, skill, and legacy.\n"
    "You MUST pick exactly one winner.\n\n"
    "Reply with ONLY the winner's full name exactly as written above. "
    "No explanation, no punctuation, no other text — just the full name."
)

CSV_HEADERS = ["matchup_id", "player_a", "player_b", "model", "result"]


# ---------------------------------------------------------------------------
# Model query
# ---------------------------------------------------------------------------
def query_judge(judge: dict, matchup_id: int, player_a: str, player_b: str) -> dict:
    user_content = PROMPT_TEMPLATE.format(player_a=player_a, player_b=player_b)
    payload = {
        "model": judge["model"],
        "messages": [
            SYSTEM_MESSAGE,
            {"role": "user", "content": user_content},
        ],
        "temperature": TEMPERATURE,
    }
    try:
        resp = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        pick = parse_winner(content, player_a, player_b)
    except Exception as e:
        print(f"\n  [!] {judge['name']} pair {matchup_id} error: {e}", file=sys.stderr)
        pick = None

    return {
        "matchup_id": matchup_id,
        "player_a": player_a,
        "player_b": player_b,
        "model": judge["name"],
        "result": pick if pick is not None else "FAILED",
    }


def parse_winner(response: str, player_a: str, player_b: str) -> str | None:
    resp_lower = response.lower().strip().rstrip(".")

    if resp_lower == player_a.lower():
        return player_a
    if resp_lower == player_b.lower():
        return player_b

    a_last = player_a.split()[-1].lower()
    b_last = player_b.split()[-1].lower()
    if a_last != b_last:
        a_in = a_last in resp_lower
        b_in = b_last in resp_lower
        if a_in and not b_in:
            return player_a
        if b_in and not a_in:
            return player_b

    a_first = player_a.split()[0].lower()
    b_first = player_b.split()[0].lower()
    if a_first != b_first:
        a_in = a_first in resp_lower
        b_in = b_first in resp_lower
        if a_in and not b_in:
            return player_a
        if b_in and not a_in:
            return player_b

    a_in = player_a.lower() in resp_lower
    b_in = player_b.lower() in resp_lower
    if a_in and not b_in:
        return player_a
    if b_in and not a_in:
        return player_b

    print(f"\n  [?] Could not parse: {response!r}", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------
def init_csv():
    if not CSV_FILE.exists():
        with open(CSV_FILE, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADERS)


def append_csv_rows(rows: list[dict]):
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        for r in sorted(rows, key=lambda x: x["matchup_id"]):
            writer.writerow([r["matchup_id"], r["player_a"], r["player_b"], r["model"], r["result"]])


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------
def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed_judges": []}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


# ---------------------------------------------------------------------------
# Run all matchups for one judge
# ---------------------------------------------------------------------------
def run_judge(judge: dict, matchups: list[tuple[int, str, str]]):
    name = judge["name"]
    total = len(matchups)
    completed = 0
    failed = 0
    t0 = time.time()

    sys.stdout.write(f"  {name}: 0/{total}...")
    sys.stdout.flush()

    results = []
    with ThreadPoolExecutor(max_workers=PARALLEL_PER_MODEL) as pool:
        futures = {
            pool.submit(query_judge, judge, mid, pa, pb): mid
            for mid, pa, pb in matchups
        }
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            completed += 1
            if result["result"] == "FAILED":
                failed += 1

            if completed % 50 == 0 or completed == total:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else 0
                sys.stdout.write(
                    f"\r  {name}: {completed}/{total} "
                    f"({failed} failed, {rate:.1f}/s, ETA {eta:.0f}s)      "
                )
                sys.stdout.flush()

    elapsed = time.time() - t0
    print(f"\r  {name}: {total}/{total} done in {elapsed:.0f}s "
          f"({failed} failed, {total/elapsed:.1f}/s)              ")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    with open(PLAYERS_FILE) as f:
        players = json.load(f)

    # Generate all pairs
    all_pairs = list(itertools.combinations(players, 2))
    matchups = [(i, a, b) for i, (a, b) in enumerate(all_pairs)]
    total_pairs = len(matchups)

    print(f"{len(players)} players, {total_pairs} pairs, {len(JUDGES)} judges")
    print(f"Total comparisons: {total_pairs * len(JUDGES):,}")
    print(f"\nJudges:")
    for j in JUDGES:
        print(f"  - {j['name']}")
    print(f"\nOllama: {OLLAMA_BASE}")
    print()

    init_csv()
    progress = load_progress()
    done_judges = set(progress.get("completed_judges", []))

    for judge in JUDGES:
        if judge["name"] in done_judges:
            print(f"  {judge['name']}: already done, skipping")
            continue

        results = run_judge(judge, matchups)
        append_csv_rows(results)

        done_judges.add(judge["name"])
        progress["completed_judges"] = list(done_judges)
        save_progress(progress)

    print(f"\nDone! All {total_pairs} pairs evaluated by all {len(JUDGES)} judges.")
    print(f"CSV: {CSV_FILE} ({total_pairs * len(JUDGES)} rows)")
    print(f"Run: python analyze.py")


if __name__ == "__main__":
    main()
