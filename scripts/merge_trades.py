"""Merge two trades.json files, preserving all unique trades from both versions.
Used in CI to prevent race-condition data loss when concurrent bot runs overwrite
each other's commits.

Usage: python3 scripts/merge_trades.py <bot_output_dir>

Never exits with non-zero status — failure is silent (main file unchanged).
"""
import json, sys, os

bot_dir   = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bot-output"
bot_path  = os.path.join(bot_dir, "trades.json")
main_path = "docs/trades.json"

try:
    with open(bot_path)  as f: bot_raw  = f.read()
    with open(main_path) as f: main_raw = f.read()
except Exception as e:
    print(f"Merge read error: {e}", flush=True)
    sys.exit(0)  # Leave main unchanged — no fallback cp

try:
    bot  = json.loads(bot_raw)
    main = json.loads(main_raw)
except Exception as e:
    print(f"Merge JSON parse error: {e}", flush=True)
    sys.exit(0)  # Leave main unchanged — no fallback cp

# Union of all trades deduped by (time, ticker, action)
seen = {}
for t in main.get("trades", []):
    k = (t.get("time",""), t.get("ticker",""), t.get("action",""))
    seen[k] = t
for t in bot.get("trades", []):
    k = (t.get("time",""), t.get("ticker",""), t.get("action",""))
    seen[k] = t  # bot's version wins on conflict

merged_trades = sorted(seen.values(), key=lambda x: x.get("time",""))

# Sanity check: merged must not have fewer trades than either input
main_count = len(main.get("trades", []))
bot_count  = len(bot.get("trades", []))
if len(merged_trades) < max(main_count, bot_count):
    print(f"Merge sanity FAIL: {len(merged_trades)} < max({main_count},{bot_count}) — keeping larger version", flush=True)
    # Keep whichever has more trades
    if main_count >= bot_count:
        print(f"Keeping main ({main_count} trades)", flush=True)
        sys.exit(0)
    else:
        # Write bot version as-is
        result = dict(bot)
        result["trades"] = sorted(bot.get("trades",[]), key=lambda x: x.get("time",""))
        try:
            with open(main_path, "w") as f:
                json.dump(result, f)
            print(f"Wrote bot version: {bot_count} trades", flush=True)
        except Exception as e:
            print(f"Write error: {e}", flush=True)
        sys.exit(0)

# Use bot's metadata (more recent state), merged trade list
result = dict(bot)
result["trades"] = merged_trades

# Merge neuron perf dicts: learning counters are cumulative, so a stale bot run
# must never regress buckets that main has already advanced. Per bucket, keep
# whichever version has more observations.
def _obs(x):
    return (x.get("total", x.get("trades", 0)) or 0) if isinstance(x, dict) else -1

for k in set(main) | set(bot):
    if not k.endswith("_perf"):
        continue
    mv, bv = main.get(k), bot.get(k)
    if isinstance(mv, dict) and isinstance(bv, dict):
        merged_perf = {}
        for bk in set(mv) | set(bv):
            a, b = mv.get(bk), bv.get(bk)
            merged_perf[bk] = a if _obs(a) > _obs(b) else b
        result[k] = merged_perf
    elif isinstance(mv, dict) and bv is None:
        result[k] = mv  # main has learning the bot version lacks entirely

try:
    with open(main_path, "w") as f:
        json.dump(result, f)
except Exception as e:
    print(f"Merge write error: {e}", flush=True)
    sys.exit(0)

print(f"Merged: main={main_count} bot={bot_count} result={len(merged_trades)}", flush=True)
