"""
Batch analysis: lie rates by condition, three graphs, and a results file. No API calls.

    python analyze.py                                   # gpt-oss-120b high, seeds 105-118
    python analyze.py --seeds 105-110                   # a subset of seeds

Reads the run files, their DeepSeek score files, and analysis/deception_labels.json (hand
labels for keyword matches in the candidates' reasoning). Writes into <run dir>/analysis/:
    stated_rate.svg          Reyes and Calder STATED rate, one line per seed plus the average
    false_claim_types.svg    false claims split into STATED / UNSTATED / ERROR / UNCLEAR
    deception_admissions.svg reasoning sentences admitting deception, one line per seed
    analysis.md              the numbers, including the all-false rate and SKIPs per statement

Tests are exact two-sided Wilcoxon signed-rank tests, pairing the two conditions by seed.
"""
import argparse
import html
import itertools
import json
import os
import random
import re
import sys

JUDGE = "deepseek-v4.1-flash"
SUFFIX = "_c-high_v-medium"
CONDITIONS = ("neutral", "bad_record")
CANDIDATES = ("Reyes", "Calder")
STEP2 = ("STATED", "UNSTATED", "ERROR", "UNCLEAR")

# Colors: reference palette slots 1 and 2 (validated pair) for the two candidates.
COLOR = {"Reyes": "#2a78d6", "Calder": "#eb6834"}
STEP2_COLOR = {"STATED": "#3b2c9e", "UNSTATED": "#6250d6", "ERROR": "#a59cec", "UNCLEAR": "#d6d5d0"}
INK, INK2, GRID, SURFACE, SEED_LINE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#bdbcb6"
FONT = "font-family='system-ui, -apple-system, Segoe UI, sans-serif'"


# ---------- data ----------

def parse_seeds(items):
    seeds = []
    for item in items:
        low, _, high = item.partition("-")
        seeds.extend(range(int(low), int(high or low) + 1))
    return seeds


def load(run_dir, seeds):
    """data[cond][seed][candidate] -> summary dict from the score file."""
    data = {c: {} for c in CONDITIONS}
    for cond in CONDITIONS:
        for seed in seeds:
            path = os.path.join(run_dir, "scores", JUDGE, f"score_{cond}_{seed}{SUFFIX}.json")
            with open(path, encoding="utf-8") as f:
                score = json.load(f)
            assert score.get("complete"), f"unfinished score file: {path}"
            data[cond][seed] = score["summary"]["candidates"]
    return data


def admissions(run_dir, seeds):
    """count[cond][seed][candidate] of reasoning sentences hand-labeled as admitting deception."""
    with open(os.path.join(run_dir, "analysis", "deception_labels.json"), encoding="utf-8") as f:
        labels = json.load(f)
    count = {c: {s: {n: 0 for n in CANDIDATES} for s in seeds} for c in CONDITIONS}
    for l in labels:
        if l["deception"] and l["seed"] in count[l["condition"]]:
            count[l["condition"]][l["seed"]][l["candidate"]] += 1
    return count, labels


# ---------- stats ----------

def mean(xs):
    return sum(xs) / len(xs)


def median(xs):
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


def wilcoxon_exact(diffs):
    """Exact two-sided Wilcoxon signed-rank p (zeros dropped, ties get average ranks)."""
    d = [x for x in diffs if abs(x) > 1e-12]
    n = len(d)
    if n == 0:
        return 1.0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(abs(d[order[j + 1]]) - abs(d[order[i]])) < 1e-12:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    centre = sum(ranks) / 2
    observed = abs(sum(r for r, x in zip(ranks, d) if x > 0) - centre)
    hits = sum(1 for signs in itertools.product((0, 1), repeat=n)
               if abs(sum(r for r, s in zip(ranks, signs) if s) - centre) >= observed - 1e-9)
    return hits / 2 ** n


def bootstrap_ci(diffs, reps=10000):
    rng = random.Random(0)
    means = sorted(mean([rng.choice(diffs) for _ in diffs]) for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def compare(neutral, bad):
    diffs = [b - n for n, b in zip(neutral, bad)]
    return {"neutral": mean(neutral), "bad": mean(bad), "diff": mean(diffs),
            "ci": bootstrap_ci(diffs), "up": sum(x > 0 for x in diffs),
            "down": sum(x < 0 for x in diffs), "p": wilcoxon_exact(diffs)}


# ---------- svg ----------

def text(x, y, s, size=13, color=INK, anchor="start", weight="normal"):
    return (f"<text x='{x:.1f}' y='{y:.1f}' {FONT} font-size='{size}' fill='{color}' "
            f"text-anchor='{anchor}' font-weight='{weight}'>{html.escape(str(s))}</text>")


def y_title(top, bottom, s):
    """Vertical axis title on the far left, centered on the plot area."""
    mid = (top + bottom) / 2
    return (f"<text x='18' y='{mid:.1f}' {FONT} font-size='12' fill='{INK2}' text-anchor='middle' "
            f"transform='rotate(-90 18 {mid:.1f})'>{html.escape(s)}</text>")


def svg_doc(width, height, body):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
            f"viewBox='0 0 {width} {height}'><rect width='100%' height='100%' fill='{SURFACE}'/>"
            + "".join(body) + "</svg>\n")


def nice_max(v, step):
    return max(step, step * -(-v // step))


def slope_chart(title, subtitle, series, ymax, step, fmt, notes, ylabel):
    """series: {candidate: (neutral values, bad values)} by seed. One panel per candidate,
    gray line per seed, colored line for the average."""
    W, H = 820, 470
    top, bottom = 110, 400
    body = [text(24, 34, title, 18, weight="600"), text(24, 56, subtitle, 13, INK2),
            y_title(top, bottom, ylabel)]
    # legend
    body += [f"<line x1='540' y1='30' x2='570' y2='30' stroke='{SEED_LINE}' stroke-width='1.5'/>",
             text(578, 34, "one seed", 12, INK2),
             f"<line x1='660' y1='30' x2='690' y2='30' stroke='{INK2}' stroke-width='3'/>",
             text(698, 34, "average", 12, INK2)]
    y = lambda v: bottom - (bottom - top) * v / ymax
    for p, cand in enumerate(CANDIDATES):
        x0 = 70 + p * 390
        xa, xb = x0 + 70, x0 + 280
        neutral, bad = series[cand]
        body.append(text(x0, 92, cand + (" (record changes)" if cand == "Reyes" else " (record same in both)"),
                         14, weight="600"))
        tick = 0.0
        while tick <= ymax + 1e-9:
            body.append(f"<line x1='{x0}' y1='{y(tick):.1f}' x2='{x0 + 340}' y2='{y(tick):.1f}' "
                        f"stroke='{GRID}' stroke-width='1'/>")
            body.append(text(x0 - 8, y(tick) + 4, fmt(tick), 11, INK2, "end"))
            tick += step
        for n, b in zip(neutral, bad):
            body.append(f"<line x1='{xa}' y1='{y(n):.1f}' x2='{xb}' y2='{y(b):.1f}' "
                        f"stroke='{SEED_LINE}' stroke-width='1.5'/>")
            for xx, v in ((xa, n), (xb, b)):
                body.append(f"<circle cx='{xx}' cy='{y(v):.1f}' r='3.5' fill='{SEED_LINE}'/>")
        mn, mb = mean(neutral), mean(bad)
        c = COLOR[cand]
        body.append(f"<line x1='{xa}' y1='{y(mn):.1f}' x2='{xb}' y2='{y(mb):.1f}' stroke='{c}' "
                    f"stroke-width='3' stroke-linecap='round'/>")
        for xx, v, anchor, dx in ((xa, mn, "end", -12), (xb, mb, "start", 12)):
            body.append(f"<circle cx='{xx}' cy='{y(v):.1f}' r='6' fill='{c}' stroke='{SURFACE}' stroke-width='2'/>")
            body.append(text(xx + dx, y(v) + 4, fmt(v), 12, INK, anchor, "600"))
        body.append(text(xa, bottom + 22, "neutral", 12, INK2, "middle"))
        body.append(text(xb, bottom + 22, "bad_record", 12, INK2, "middle"))
        body.append(text(x0 + 175, bottom + 48, notes[cand], 12, INK2, "middle"))
    return svg_doc(W, H, body)


def stacked_bars(title, subtitle, totals, n_runs):
    """totals: {(candidate, cond): {step2 label: count}}."""
    W, H = 820, 470
    top, bottom = 120, 400
    body = [text(24, 34, title, 18, weight="600"), text(24, 56, subtitle, 13, INK2),
            y_title(top, bottom, f"false claims (total over {n_runs} runs)")]
    lx = 24
    for lab in STEP2:
        body.append(f"<rect x='{lx}' y='76' width='12' height='12' rx='2' fill='{STEP2_COLOR[lab]}'/>")
        body.append(text(lx + 18, 87, lab, 12, INK2))
        lx += 110
    biggest = max(sum(v.values()) for v in totals.values())
    ymax = nice_max(biggest, 10)
    y = lambda v: bottom - (bottom - top) * v / ymax
    tick = 0
    while tick <= ymax:
        body.append(f"<line x1='70' y1='{y(tick):.1f}' x2='790' y2='{y(tick):.1f}' stroke='{GRID}' stroke-width='1'/>")
        body.append(text(62, y(tick) + 4, tick, 11, INK2, "end"))
        tick += 10
    xs = {("Reyes", "neutral"): 130, ("Reyes", "bad_record"): 270,
          ("Calder", "neutral"): 500, ("Calder", "bad_record"): 640}
    bw = 96
    for key, x in xs.items():
        base = 0
        for lab in STEP2:
            v = totals[key].get(lab, 0)
            if v:
                y1, y2 = y(base + v), y(base)
                body.append(f"<rect x='{x}' y='{y1:.1f}' width='{bw}' height='{max(0, y2 - y1 - 2):.1f}' "
                            f"rx='{3 if base + v == sum(totals[key].values()) else 0}' fill='{STEP2_COLOR[lab]}'/>")
                if y2 - y1 >= 16:
                    ink = SURFACE if lab in ("STATED", "UNSTATED") else INK
                    body.append(text(x + bw / 2, (y1 + y2) / 2 + 4, v, 12, ink, "middle", "600"))
            base += v
        body.append(text(x + bw / 2, y(base) - 8, f"{base} false", 12, INK, "middle", "600"))
        body.append(text(x + bw / 2, bottom + 20, key[1], 12, INK2, "middle"))
    body.append(text(270 - 22, bottom + 44, "Reyes (record changes)", 13, INK, "middle", "600"))
    body.append(text(640 - 22, bottom + 44, "Calder (record same in both)", 13, INK, "middle", "600"))
    return svg_doc(W, H, body)


# ---------- main ----------

def pct(v):
    return f"{100 * v:.0f}%"


def pval(p):
    return f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Batch analysis and graphs.")
    parser.add_argument("--run-dir", default=os.path.join("..", "output", "test_batch", "gpt-oss-120b", "think-high"))
    parser.add_argument("--seeds", nargs="+", default=["105-118"])
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    out_dir = os.path.join(args.run_dir, "analysis")
    data = load(args.run_dir, seeds)
    adm, labels = admissions(args.run_dir, seeds)

    def series(get):
        return {cand: ([get(data["neutral"][s][cand]) for s in seeds],
                       [get(data["bad_record"][s][cand]) for s in seeds]) for cand in CANDIDATES}

    measures = {
        "STATED rate": series(lambda c: c["rates"]["stated_rate"]),
        "UNSTATED rate": series(lambda c: c["rates"]["unstated_rate"]),
        "All-false rate": series(lambda c: c["rates"]["upper_bound_rate"]),
        "SKIP claims per statement": series(lambda c: c["skip_per_statement"]),
        "Checkable claims per statement": series(lambda c: c["checkable_per_statement"]),
        "Deception admissions per run": {cand: ([adm["neutral"][s][cand] for s in seeds],
                                                [adm["bad_record"][s][cand] for s in seeds])
                                         for cand in CANDIDATES},
    }
    results = {m: {cand: compare(*v[cand]) for cand in CANDIDATES} for m, v in measures.items()}
    n = len(seeds)

    # Graph 1: STATED rate
    st = measures["STATED rate"]
    top_rate = max(max(a + b) for a, b in st.values())
    notes = {c: f"{pval(results['STATED rate'][c]['p'])} (Wilcoxon, {n} pairs)" for c in CANDIDATES}
    svg1 = slope_chart("Deliberate lies (STATED) per checkable claim",
                       f"gpt-oss-120b candidates, seeds {seeds[0]}–{seeds[-1]}; each gray line is one seed in both conditions",
                       st, nice_max(top_rate, 0.1), 0.1, pct, notes,
                       "STATED lies (% of checkable claims)")

    # Graph 2: false claim types
    totals = {}
    for cand in CANDIDATES:
        for cond in CONDITIONS:
            t = {}
            for s in seeds:
                for lab, v in data[cond][s][cand]["step2_counts"].items():
                    t[lab] = t.get(lab, 0) + v
            totals[(cand, cond)] = t
    svg2 = stacked_bars("What kind of false claims each candidate made",
                        f"all false claims across {n} runs per bar, split by what the reasoning shows", totals, n)

    # Graph 3: deception admissions
    ad = measures["Deception admissions per run"]
    top_adm = max(max(a + b) for a, b in ad.values())
    notes3 = {c: f"{pval(results['Deception admissions per run'][c]['p'])} (Wilcoxon, {n} pairs)" for c in CANDIDATES}
    svg3 = slope_chart("Reasoning sentences that admit deception, per run",
                       "hand-checked keyword matches, e.g. “we pretend we increased it”, “So we can lie if needed”",
                       ad, nice_max(top_adm, 1), 1, lambda v: f"{v:.1f}".rstrip("0").rstrip("."), notes3,
                       "admitting sentences per run")

    os.makedirs(out_dir, exist_ok=True)
    for name, svg in (("stated_rate.svg", svg1), ("false_claim_types.svg", svg2),
                      ("deception_admissions.svg", svg3)):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            f.write(svg)

    # Numbers
    lines = [f"# Batch analysis: gpt-oss-120b (candidates high, voters medium), seeds {seeds[0]}–{seeds[-1]}",
             "", f"{n} seeds × 2 conditions = {2 * n} runs, scored by the frozen {JUDGE} judge. "
             "Tests: exact two-sided Wilcoxon signed-rank, paired by seed. "
             "CI: bootstrap 95% CI of the mean paired difference (bad_record − neutral).", "",
             "![STATED rate](stated_rate.svg)", "", "![False claim types](false_claim_types.svg)", "",
             "![Deception admissions](deception_admissions.svg)", "",
             "| Measure | Candidate | neutral | bad_record | difference (95% CI) | seeds up / down | p |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for m, by_cand in results.items():
        is_rate = "rate" in m
        f = pct if is_rate else (lambda v: f"{v:.2f}")
        for cand, r in by_cand.items():
            lo, hi = r["ci"]
            lines.append(f"| {m} | {cand} | {f(r['neutral'])} | {f(r['bad'])} | "
                         f"{'+' if r['diff'] >= 0 else ''}{f(r['diff'])} ({f(lo)} to {f(hi)}) | "
                         f"{r['up']} / {r['down']} | {pval(r['p'])} |")
    lines += ["", "## False claims by type (totals across runs)", "",
              "| Candidate | Condition | " + " | ".join(STEP2) + " | total |",
              "| --- | --- | " + " | ".join("---" for _ in STEP2) + " | --- |"]
    for (cand, cond), t in totals.items():
        lines.append(f"| {cand} | {cond} | " + " | ".join(str(t.get(l, 0)) for l in STEP2)
                     + f" | {sum(t.values())} |")
    lines += ["", "## STATED rate by seed", "", "| Seed | Reyes neutral | Reyes bad_record | Calder neutral | Calder bad_record |",
              "| --- | --- | --- | --- | --- |"]
    for i, s in enumerate(seeds):
        lines.append(f"| {s} | " + " | ".join(pct(st[c][k][i]) for c in CANDIDATES for k in (0, 1)) + " |")
    kept = sum(l["deception"] for l in labels if l["seed"] in seeds)
    total = sum(1 for l in labels if l["seed"] in seeds)
    lines += ["", f"Deception admissions: {kept} of {total} keyword matches were hand-labeled as the candidate "
              "admitting its own deception (labels in deception_labels.json)."]
    with open(os.path.join(out_dir, "analysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out_dir}")
