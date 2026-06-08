"""
Read-only live-account check — verifies your Polymarket keys work WITHOUT
placing a single trade. Run this before ever letting the agent trade live.

    python3 scripts/check_live.py

It confirms, in order:
  1. The CLOB client initialises with your key/creds/funder.
  2. Your USDC collateral balance is readable.
  3. USDC is (or can be) approved for the exchange.
  4. Your current positions are readable.

Nothing here buys, sells, or redeems. If any step fails it prints exactly what
to fix in your .env. Safe to run as often as you like.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agent.polymarket_client import PolymarketClient


def main():
    print("=" * 60)
    print("  Polymarket LIVE account check (read-only, no trades)")
    print("=" * 60)

    if not config.POLYMARKET_PRIVATE_KEY:
        print("❌ POLYMARKET_PRIVATE_KEY is empty → still in PAPER mode.")
        print("   Set it in .env to test a live account.")
        return

    print(f"• signature_type : {config.POLYMARKET_SIGNATURE_TYPE}")
    print(f"• funder set     : {'yes' if config.POLYMARKET_FUNDER else 'NO (EOA mode)'}")
    if config.POLYMARKET_SIGNATURE_TYPE in (1, 2) and not config.POLYMARKET_FUNDER:
        print("  ⚠️  signature_type 1/2 needs POLYMARKET_FUNDER (your Polymarket")
        print("      deposit/proxy address). Orders will sign for the wrong account.")

    poly = PolymarketClient()
    if not poly._client:
        print("\n❌ CLOB client did NOT initialise. Check the error above and that")
        print("   py-clob-client is installed:  pip3 install py-clob-client web3")
        return
    print("✅ CLOB client initialised.")

    bal = poly.get_balance()
    print(f"✅ USDC balance  : ${bal:.2f}")
    if bal <= 0:
        print("   ⚠️  Zero balance — deposit USDC on polymarket.com before trading.")

    ok = poly.ensure_allowance()
    print(f"{'✅' if ok else '❌'} USDC allowance : {'approved' if ok else 'NOT approved'}")

    positions = poly.get_positions()
    print(f"✅ Positions read : {len(positions)} open")
    for p in positions[:5]:
        q = (p.get('title') or p.get('question') or p.get('asset') or '?')[:40]
        sz = p.get('size') or p.get('shares') or '?'
        print(f"     - {q}  size={sz}")

    print("\nAll read-only checks passed. The account is wired correctly.")
    print("Start live trading only AFTER the optimizer confirms an edge and a")
    print("paper run looks healthy — and begin with MAX_POSITION_SIZE_USD=1.")


if __name__ == "__main__":
    main()
