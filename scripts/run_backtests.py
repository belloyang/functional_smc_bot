#!/usr/bin/env python3
"""Batch runner for scripts/backtest.py.

Supports two ways to define runs:
1) Config file (JSON)
2) CLI variables (--symbols/--days/--balances/--modes)

Outputs:
- Backtest charts under a dedicated output folder (default: backtest-runs/<timestamp>/backtest-output/...)
- Run logs under <output>/logs/
"""

from __future__ import annotations

import os
import argparse
import itertools
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from app.config import load_env


VALID_MODES = {"stock", "options"}


@dataclass
class BacktestJob:
    symbol: str
    days: int
    balance: float
    mode: str
    cap: int | None = None
    stock_budget: float | None = None
    option_budget: float | None = None
    min_conf: str | None = None

    def id_slug(self) -> str:
        return f"{self.symbol}_{self.mode}_{self.days}d_{self.balance:g}".replace("/", "_")


def parse_csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def load_config(path: Path | None) -> dict[str, Any]:
    if not path:
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a JSON object")
    return data


def build_jobs(args: argparse.Namespace, cfg: dict[str, Any]) -> list[BacktestJob]:
    jobs_raw = cfg.get("jobs")
    common = cfg.get("common", {}) if isinstance(cfg.get("common", {}), dict) else {}

    jobs: list[BacktestJob] = []

    if isinstance(jobs_raw, list) and jobs_raw:
        for j in jobs_raw:
            if not isinstance(j, dict):
                raise ValueError("Each item in config.jobs must be an object")
            mode = str(j.get("mode", common.get("mode", "stock"))).lower()
            if mode not in VALID_MODES:
                raise ValueError(f"Invalid mode in jobs: {mode}")
            jobs.append(
                BacktestJob(
                    symbol=str(j.get("symbol", common.get("symbol", "SPY"))).upper(),
                    days=int(j.get("days", common.get("days", 30))),
                    balance=float(j.get("balance", common.get("balance", 10000))),
                    mode=mode,
                    cap=(None if j.get("cap", common.get("cap")) is None else int(j.get("cap", common.get("cap")))),
                    stock_budget=(None if j.get("stock_budget", common.get("stock_budget")) is None else float(j.get("stock_budget", common.get("stock_budget")))),
                    option_budget=(None if j.get("option_budget", common.get("option_budget")) is None else float(j.get("option_budget", common.get("option_budget")))),
                    min_conf=(None if j.get("min_conf", common.get("min_conf")) is None else str(j.get("min_conf", common.get("min_conf")))),
                )
            )
        return jobs

    # CLI/config cartesian mode
    symbols = parse_csv_list(args.symbols) or cfg.get("symbols") or ["SPY"]
    days = parse_csv_list(args.days) or cfg.get("days") or [30]
    balances = parse_csv_list(args.balances) or cfg.get("balances") or [10000]
    modes = [m.lower() for m in (parse_csv_list(args.modes) or cfg.get("modes") or ["stock"])]

    for m in modes:
        if m not in VALID_MODES:
            raise ValueError(f"Invalid mode: {m}. Use one of {sorted(VALID_MODES)}")

    cap = args.cap if args.cap is not None else cfg.get("cap")
    stock_budget = args.stock_budget if args.stock_budget is not None else cfg.get("stock_budget")
    option_budget = args.option_budget if args.option_budget is not None else cfg.get("option_budget")
    min_conf = args.min_conf if args.min_conf is not None else cfg.get("min_conf")

    for symbol, day, bal, mode in itertools.product(symbols, days, balances, modes):
        jobs.append(
            BacktestJob(
                symbol=str(symbol).upper(),
                days=int(day),
                balance=float(bal),
                mode=mode,
                cap=(None if cap is None else int(cap)),
                stock_budget=(None if stock_budget is None else float(stock_budget)),
                option_budget=(None if option_budget is None else float(option_budget)),
                min_conf=(None if min_conf is None else str(min_conf)),
            )
        )

    return jobs


def build_command(python_bin: str, backtest_script: Path, job: BacktestJob) -> list[str]:
    cmd = [python_bin, str(backtest_script), job.symbol, "--days", str(job.days), "--balance", str(job.balance)]

    if job.mode == "options":
        cmd.append("--options")

    if job.cap is not None:
        cmd.extend(["--cap", str(job.cap)])
    if job.stock_budget is not None:
        cmd.extend(["--stock-budget", str(job.stock_budget)])
    if job.option_budget is not None:
        cmd.extend(["--option-budget", str(job.option_budget)])
    if job.min_conf is not None:
        cmd.extend(["--min-conf", str(job.min_conf)])

    return cmd


def main() -> int:
    # Load environment variables from .env if it exists
    load_env()
    parser = argparse.ArgumentParser(description="Run many backtests with configurable parameters.")
    parser.add_argument("--config", type=str, help="Path to JSON config file.")
    parser.add_argument("--symbols", type=str, help="CSV symbols, e.g. SPY,QQQ,SLV")
    parser.add_argument("--days", type=str, help="CSV days, e.g. 30,90,365")
    parser.add_argument("--balances", type=str, help="CSV initial balances, e.g. 2500,10000")
    parser.add_argument("--modes", type=str, help="CSV modes: stock,options")
    parser.add_argument("--cap", type=int, help="Daily trade cap (-1 for unlimited)")
    parser.add_argument("--stock-budget", type=float, help="Override stock allocation pct")
    parser.add_argument("--option-budget", type=float, help="Override option allocation pct")
    parser.add_argument("--min-conf", type=str, choices=["all", "low", "medium", "high"], help="Signal confidence filter")
    parser.add_argument("--output-dir", type=str, help="Dedicated folder for this batch run")
    parser.add_argument("--python-bin", type=str, default="python3.12", help="Python binary for backtest command")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    backtest_script = repo_root / "scripts" / "backtest.py"

    cfg_path = Path(args.config).resolve() if args.config else None
    cfg = load_config(cfg_path)
    jobs = build_jobs(args, cfg)

    if not jobs:
        print("No jobs to run.")
        return 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_raw = args.output_dir or cfg.get("output_dir") or f"backtest-runs/{stamp}"
    output_dir = Path(output_dir_raw).expanduser()
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output folder: {output_dir}")
    print(f"Total jobs: {len(jobs)}")

    failures = 0
    for idx, job in enumerate(jobs, start=1):
        cmd = build_command(args.python_bin, backtest_script, job)
        print(f"[{idx}/{len(jobs)}] {job.id_slug()}")
        print(" ", " ".join(cmd))

        if args.dry_run:
            continue

        log_path = logs_dir / f"{idx:03d}_{job.id_slug()}.log"
        with log_path.open("w", encoding="utf-8") as logf:
            proc = subprocess.run(
                cmd,
                cwd=str(output_dir),
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )

        if proc.returncode != 0:
            failures += 1
            print(f"  -> FAIL (exit={proc.returncode}) log={log_path}")
        else:
            print(f"  -> PASS log={log_path}")

    print("-" * 60)
    print(f"Finished. failures={failures}, success={len(jobs) - failures}, total={len(jobs)}")
    print(f"Charts folder base: {output_dir / 'backtest-output'}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
