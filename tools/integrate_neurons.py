#!/usr/bin/env python3
"""
ROK neuron integration helper.

After a batch of neurons (e.g. N421-N430) is added to rok_trader.py by an agent,
this script wires them into the dashboard + intel pipeline:
  1. generate_report.py  -> neuron_map label entries (for weekly report)
  2. docs/index.html     -> NEURON_CATALOG entries (for the Brain Map page)
  3. docs/sw.js          -> cache version bump (forces clients to refresh)
  4. docs/trades.json    -> neurons_total (left to the live bot; only bumped if requested)

Idempotent: skips any neuron id/key already present. Safe to re-run.

Usage:
  python3 tools/integrate_neurons.py <batchfile.json>

batchfile.json format:
  {
    "neurons": [
      {"id":"N421","key":"atr_expansion_entry_perf","name":"ATR Expansion Entry",
       "cat":"Technical","desc":"ATR expanding vs contracting at entry","label":"N421 ATR Expansion Entry"}
    ],
    "bump_cache": true
  }
"""
import json, re, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, "generate_report.py")
HTML = os.path.join(ROOT, "docs", "index.html")
SW = os.path.join(ROOT, "docs", "sw.js")


def integrate_generate_report(neurons):
    src = open(GEN, encoding="utf-8").read()
    added = []
    # Find the neuron_map dict closing — anchor on the last existing N4xx entry then closing brace.
    # We insert new entries right before the line that closes the neuron_map ("    }\n    for key, label").
    m = re.search(r'(\n)(    \}\n    for key, label in neuron_map\.items\(\):)', src)
    if not m:
        return "generate_report: ANCHOR_NOT_FOUND"
    insert_lines = []
    for n in neurons:
        key = n["key"]
        if f'"{key}"' in src:
            continue
        label = n.get("label", f'{n["id"]} {n["name"]}')
        insert_lines.append(f'        "{key}": "{label}",')
        added.append(n["id"])
    if not insert_lines:
        return "generate_report: nothing new"
    block = "\n".join(insert_lines) + "\n"
    src = src[:m.start(2)] + block + src[m.start(2):]
    open(GEN, "w", encoding="utf-8").write(src)
    return "generate_report: added " + ",".join(added)


def integrate_html(neurons):
    src = open(HTML, encoding="utf-8").read()
    # NEURON_CATALOG is an array ending with "  ];" after the last {id:...} entry.
    m = re.search(r'(\n)(  \];\n)', src[src.index("const NEURON_CATALOG"):])
    if not m:
        return "html: ANCHOR_NOT_FOUND"
    base = src.index("const NEURON_CATALOG")
    abs_close = base + m.start(2)
    insert_lines = []
    added = []
    for n in neurons:
        if f"id:'{n['id']}'" in src or f'id:"{n["id"]}"' in src:
            continue
        desc = n["desc"].replace("'", "\\'")
        name = n["name"].replace("'", "\\'")
        insert_lines.append(
            f"    {{id:'{n['id']}',key:'{n['key']}',name:'{name}',cat:'{n['cat']}',desc:'{desc}'}},"
        )
        added.append(n["id"])
    if not insert_lines:
        return "html: nothing new"
    block = "\n".join(insert_lines) + "\n"
    src = src[:abs_close] + block + src[abs_close:]
    open(HTML, "w", encoding="utf-8").write(src)
    return "html: added " + ",".join(added)


def bump_cache():
    src = open(SW, encoding="utf-8").read()
    m = re.search(r"const CACHE = 'rok-v(\d+)';", src)
    if not m:
        return "sw: ANCHOR_NOT_FOUND"
    old = int(m.group(1))
    new = old + 1
    src = src.replace(f"rok-v{old}", f"rok-v{new}")
    open(SW, "w", encoding="utf-8").write(src)
    return f"sw: bumped v{old} -> v{new}"


def main():
    if len(sys.argv) < 2:
        print("usage: integrate_neurons.py <batch.json>")
        sys.exit(1)
    batch = json.load(open(sys.argv[1]))
    neurons = batch["neurons"]
    results = []
    results.append(integrate_generate_report(neurons))
    results.append(integrate_html(neurons))
    if batch.get("bump_cache", True):
        results.append(bump_cache())
    print(" || ".join(results))


if __name__ == "__main__":
    main()
