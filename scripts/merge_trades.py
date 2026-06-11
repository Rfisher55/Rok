"""Merge two trades.json files, preserving all unique trades from both versions.
Used in CI to prevent race-condition data loss when concurrent bot runs overwrite
each other's commits.

Usage: python3 scripts/merge_trades.py <bot_output_dir>
"""
import json, sys, os

bot_dir   = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bot-output"
bot_path  = os.path.join(bot_dir, "trades.json")
main_path = "docs/trades.json"

try:
    with open(bot_path)  as f: bot  = json.load(f)
    with open(main_path) as f: main = json.load(f)
except Exception as e:
    print(f"Merge load error: {e}", flush=True)
    sys.exit(1)

# Union of all trades deduped by (time, ticker, action)
seen = {}
for t in main.get("trades", []):
    k = (t.get("time",""), t.get("ticker",""), t.get("action",""))
    seen[k] = t
for t in bot.get("trades", []):
    k = (t.get("time",""), t.get("ticker",""), t.get("action",""))
    seen[k] = t  # bot's version wins on conflict

merged_trades = sorted(seen.values(), key=lambda x: x.get("time",""))

# Use bot's metadata (more recent state), merged trade list
result = dict(bot)
result["trades"] = merged_trades

with open(main_path, "w") as f:
    json.dump(result, f)

print(f"Merged: main={len(main.get('trades',[]))} bot={len(bot.get('trades',[]))} result={len(merged_trades)}", flush=True)
