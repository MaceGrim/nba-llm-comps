# nba-llm-comps

Five small open-source LLMs each judged every head-to-head matchup between 144 NBA all-time candidates: 10,296 pairs per model, 51,473 valid votes. Elo ratings turn the votes into rankings.

Write-up with interactive charts: [Five Tiny AIs Ranked 144 NBA Legends](https://macegrim.github.io/blog/2026-05-nba-llm-rankings/)

## How it works

1. `players.json` holds the 144-player pool.
2. `elo_ranker.py` asks every judge about every pair, one model at a time (requests run in parallel within a model), and appends one row per vote to `matchups.csv`. Progress checkpoints to `progress.json`, so an interrupted run resumes where it left off.
3. `analyze.py` reconstructs everything from the CSV: a combined Elo ladder over all votes, a separate ladder per judge, win rates, and a judge-vs-judge agreement matrix.
4. `prep_viz_data.py` emits the compact JSON the blog charts consume.

Every model sees the same system message and prompt, at temperature 0.3:

> You are a sports analyst. When asked to compare players, you must pick one. No hedging.

> Who is the better NBA player: {player_a} or {player_b}?
> Consider career achievements, impact, skill, and legacy.
> You MUST pick exactly one winner.
>
> Reply with ONLY the winner's full name exactly as written above. No explanation, no punctuation, no other text — just the full name.

A reply counts as a vote only if it parses to one of the two names (51,473 of 51,480 did).

## The judges

| Model | Size |
|---|---|
| gemma3:12b | 12B |
| phi4:14b | 14B |
| llama3.1:8b | 8B |
| mistral:7b | 7B |
| qwen2.5:7b | 7B |

## Reproduce it

You need [Ollama](https://ollama.com) with the five models pulled:

```sh
for m in gemma3:12b llama3.1:8b phi4:14b mistral:7b qwen2.5:7b; do ollama pull $m; done
```

Then:

```sh
python3 elo_ranker.py     # hours of inference; resumes if interrupted
python3 analyze.py        # prints rankings, agreement matrix, controversy stats
python3 prep_viz_data.py  # writes viz_data.json + pair_votes.json
```

Ollama runs at `http://localhost:11434` by default; point `OLLAMA_BASE` elsewhere if yours lives on another machine. The only dependency outside the standard library is `requests`.

`matchups.csv` in this repo is the full raw vote data from my run, so you can skip straight to `analyze.py`, or swap in your own `players.json` (any domain where two names fit in the prompt) and run the whole thing fresh.

## Method notes

- Elo: K=32, everyone starts at 1500. Per-model ladders use only that judge's votes; the combined ladder replays every valid vote.
- Each pair was asked in one direction only. LLMs can be sensitive to which name comes first, and that bias is not measured here. Treat the rankings as a strong signal of model personality, not a precision instrument.
- Small models (7B to 14B) know less than frontier models, about basketball and everything else. That is part of the fun.
