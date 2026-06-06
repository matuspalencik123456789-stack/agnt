"""
Entry point — runs the Polymarket BTC trading agent on a 15-minute schedule.
Usage:
    python main.py              # start the trading agent
    python main.py --dashboard  # start the Streamlit dashboard
    python main.py --both       # start both in parallel
"""
import sys
import os
import logging
import colorlog
import argparse
import threading
import time

from apscheduler.schedulers.blocking import BlockingScheduler

from agent.database.models import init_db
from agent.trader import Trader
import config


def setup_logging():
    os.makedirs(os.path.dirname(config.LOG_PATH), exist_ok=True)
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        log_colors={
            "DEBUG":    "cyan",
            "INFO":     "green",
            "WARNING":  "yellow",
            "ERROR":    "red",
            "CRITICAL": "bold_red",
        }
    ))
    file_handler = logging.FileHandler(config.LOG_PATH)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    root.addHandler(file_handler)


def run_agent():
    setup_logging()
    log = logging.getLogger("main")

    is_paper = not config.POLYMARKET_PRIVATE_KEY
    log.info("=" * 64)
    log.info(f"  MODE: {'PAPER (simulated, no real money)' if is_paper else 'LIVE TRADING'}")
    log.info(f"  Trading only REAL Polymarket 15-min BTC slugs (live discovery)")
    log.info(f"  Cycle: every {config.TRADING_INTERVAL_MINUTES} min | "
             f"Roll-check: every {config.ROLL_CHECK_SECONDS}s")
    log.info("=" * 64)

    log.info("Initializing database...")
    init_db()

    trader = Trader()

    # Run immediately on start
    log.info("Running initial cycle...")
    try:
        trader.run_cycle()
        trader.check_resolutions()
    except Exception as e:
        log.error(f"Initial cycle error: {e}")

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        trader.run_cycle,
        "interval",
        minutes=config.TRADING_INTERVAL_MINUTES,
        id="trading_cycle",
        max_instances=1,
    )
    scheduler.add_job(
        trader.check_resolutions,
        "interval",
        minutes=5,
        id="resolution_check",
        max_instances=1,
    )
    # Fast hand-off: roll onto the next 15-min BTC slug the instant the current ends
    scheduler.add_job(
        trader.roll_check,
        "interval",
        seconds=config.ROLL_CHECK_SECONDS,
        id="roll_check",
        max_instances=1,
    )

    log.info(f"Scheduler started — trading every {config.TRADING_INTERVAL_MINUTES} minutes.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Agent stopped.")


def run_dashboard():
    # Use `python3 -m streamlit` so it works even when the `streamlit` console
    # script isn't on PATH (common with --user pip installs on macOS).
    import sys
    os.execv(sys.executable, [
        sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
        "--server.port", os.getenv("DASHBOARD_PORT", "8501"),
        "--server.headless", "true",
    ])


def main():
    parser = argparse.ArgumentParser(description="Polymarket BTC Trading Agent")
    parser.add_argument("--dashboard", action="store_true", help="Run dashboard only")
    parser.add_argument("--both",      action="store_true", help="Run agent + dashboard")
    args = parser.parse_args()

    if args.dashboard:
        run_dashboard()
    elif args.both:
        t = threading.Thread(target=run_agent, daemon=True)
        t.start()
        time.sleep(2)
        run_dashboard()   # blocks
    else:
        run_agent()


if __name__ == "__main__":
    main()
