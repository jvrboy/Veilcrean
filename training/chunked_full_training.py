"""Resumable full-scope Deriv trainer.

Runs the requested 15-pass training plan in small symbol batches so long jobs can
be resumed safely from repository artifacts. Each invocation trains the next
pending 1-3 instruments across all configured timeframes using real Deriv
historical candles and saves progress after the batch.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Make the repository importable no matter where this file is run from
# (module import, direct execution, or a Colab notebook).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from training.deriv_client import get_all_instruments, get_all_timeframes
from training.training_runner import run_full_training

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
PROGRESS_FILE = OUTPUT_DIR / "full_training_progress.json"


def _load_progress() -> dict[str, Any]:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    instruments = get_all_instruments()
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "status": "in_progress",
        "requested_runs_per_instrument": 15,
        "requested_timeframes": get_all_timeframes(),
        "total_instruments": len(instruments),
        "completed_symbols": [],
        "batches": [],
    }


def _save_progress(progress: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress["updated_at"] = datetime.now(UTC).isoformat()
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def _next_symbols(progress: dict[str, Any], batch_size: int) -> list[str]:
    completed = set(progress.get("completed_symbols", []))
    pending = [i["symbol"] for i in get_all_instruments() if i["symbol"] not in completed]
    return pending[:batch_size]


async def run_next_batch(batch_size: int, runs: int, history_years: float | None,
                         max_batches: int | None, quiet: bool) -> dict[str, Any]:
    if not 1 <= batch_size <= 3:
        raise ValueError("batch_size must be between 1 and 3 instruments")
    if runs < 1:
        raise ValueError("runs must be at least 1")

    progress = _load_progress()
    symbols = _next_symbols(progress, batch_size)
    if not symbols:
        progress["status"] = "complete"
        _save_progress(progress)
        return progress

    # Mutate the imported module global only when the operator deliberately caps
    # Deriv pagination for a run. Leaving it unset uses the full configured depth.
    if max_batches is not None:
        import training.training_runner as training_runner
        training_runner.MAX_BATCHES = max_batches

    started_at = datetime.now(UTC).isoformat()
    engine, summaries = await run_full_training(
        num_runs=runs,
        verbose=not quiet,
        symbols=symbols,
        timeframes_filter=get_all_timeframes(),
        resume=True,
        history_years=history_years,
    )
    finished_at = datetime.now(UTC).isoformat()

    progress = _load_progress()
    completed = list(dict.fromkeys(progress.get("completed_symbols", []) + symbols))
    progress["completed_symbols"] = completed
    progress["remaining_symbols"] = [
        i["symbol"] for i in get_all_instruments() if i["symbol"] not in set(completed)
    ]
    progress["status"] = "complete" if not progress["remaining_symbols"] else "in_progress"
    progress["batches"].append({
        "started_at": started_at,
        "finished_at": finished_at,
        "symbols": symbols,
        "timeframes": get_all_timeframes(),
        "runs": runs,
        "history_years": history_years,
        "max_batches_override": max_batches,
        "run_summaries": summaries,
        "cumulative_stats": engine.to_dict()["stats"],
    })
    _save_progress(progress)
    return progress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the next resumable full-training instrument batch.")
    parser.add_argument("--batch-size", type=int, default=3, help="Number of pending instruments to train, 1-3.")
    parser.add_argument("--runs", type=int, default=15, help="Forward-training passes per batch.")
    parser.add_argument("--history-years", type=float, default=5.0, help="Historical window to train from.")
    parser.add_argument("--max-history", action="store_true", help="Use maximum Deriv paginated history instead of --history-years.")
    parser.add_argument("--max-batches", type=int, help="Optional Deriv pagination cap for operational chunks.")
    parser.add_argument("--quiet", action="store_true", help="Reduce per-combo logging.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(run_next_batch(
            batch_size=args.batch_size,
            runs=args.runs,
            history_years=None if args.max_history else args.history_years,
            max_batches=args.max_batches,
            quiet=args.quiet,
        ))
    except ConnectionError as e:
        print(f"\n{e}", file=sys.stderr)
        raise SystemExit(1) from e
