# Signal Generator

This repository runs the signal engine locally or on GitHub Actions (cloud), so it can continue running even when your PC is off.

## What this version does

- Weekday metals gate: signals only during 04:00-18:00 in configured timezone.
- Non-metals gate: signals only if market is moving and confidence is in 0.90-0.98 by default.
- End-of-day review: evaluates prior signals and updates learning state.
- Adaptive confidence mode: learns a confidence floor from historical precision.

## Run locally

```bash
python3 signal_generator.py --eod-review --telegram --adaptive-confidence --timezone America/New_York
```

## Required GitHub Secrets

Set these in repository `Settings -> Secrets and variables -> Actions`:

- `SIGNAL_TELEGRAM_BOT_TOKEN`
- `SIGNAL_TELEGRAM_CHAT_ID`
- `FMP_API_KEY`
- `SIGNAL_TIMEZONE` (for example `America/New_York`)

## GitHub Actions

Workflow file: `.github/workflows/signal-generator.yml`

- Runs every 30 minutes.
- Runs review + learning update + signal generation.
- Commits updated `signal_journal.csv` and `learning_state.json` back to repo.

## Accuracy note

No trading model can guarantee 90-98% or 100% future accuracy in real markets.
This implementation is optimized for precision-first filtering and continuous adaptation,
but results will vary with market regime, data quality, and execution conditions.
