"""Generate the README charts from the frozen evaluation results.

    python scripts/make_readme_charts.py

Reads results/summary.json and results/one-credit.jsonl and writes PNGs into
docs/media/.  Palette and layout are fixed so regenerating keeps the README look.
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "media"

BLUE = "#2f6fed"
ORANGE = "#ef8a2b"
INK = "#1c2434"
MUTED = "#6b7789"
GRID = "#e3e8f0"
WHITE = "#ffffff"

STAGES = [
    ("ChunLi", "春丽"), ("Zangief", "桑吉尔夫"), ("Dhalsim", "达尔锡"), ("Ryu", "隆"),
    ("EHonda", "本田"), ("Blanka", "布兰卡"), ("Guile", "古烈"), ("Ken", "肯"),
    ("Balrog", "拜森"), ("Vega", "维加"), ("Sagat", "沙加特"), ("Bison", "将军"),
]
JITTERS = [("0.0", "无抖动", BLUE), ("0.05", "5% 输入抖动", ORANGE)]

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "Microsoft JhengHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.titlecolor": INK,
})


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=190, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print(f"wrote {OUT / name}")


def load_results():
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    runs = [json.loads(line) for line in (RESULTS / "one-credit.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return summary, runs


def clear_rate_chart(summary):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    style(ax)
    rates, lows, highs, labels, colors = [], [], [], [], []
    for key, label, color in JITTERS:
        block = summary["natural_campaigns"][key]
        rates.append(block["campaign_clear_rate"] * 100)
        lows.append(block["campaign_ci95"][0] * 100)
        highs.append(block["campaign_ci95"][1] * 100)
        labels.append(label)
        colors.append(color)
    bars = ax.bar(labels, rates, width=0.52, color=colors, zorder=3)
    ax.errorbar(labels, rates, yerr=[ [r - l for r, l in zip(rates, lows)], [h - r for r, h in zip(rates, highs)] ],
                fmt="none", ecolor=INK, elinewidth=1.4, capsize=5, zorder=4)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 4.2, f"{rate:.0f}%", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color=INK)
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_title("自然赛程一币通关率（100 次）", fontsize=12, pad=10)
    ax.set_ylabel("通关率")

    ax = axes[1]
    style(ax)
    rates, lows, highs, labels = [], [], [], []
    for key, label, color in JITTERS:
        block = summary["independent_fixtures"][key]
        rates.append(block["fixture_win"] * 100)
        lows.append(block["fixture_ci95"][0] * 100)
        highs.append(block["fixture_ci95"][1] * 100)
        labels.append(label)
    bars = ax.bar(labels, rates, width=0.52, color=[B["color"] for B in [dict(color=BLUE), dict(color=ORANGE)]], zorder=3)
    ax.errorbar(labels, rates, yerr=[[r - l for r, l in zip(rates, lows)], [h - r for r, h in zip(rates, highs)]],
                fmt="none", ecolor=INK, elinewidth=1.4, capsize=5, zorder=4)
    for bar, rate, key in zip(bars, rates, ("0.0", "0.05")):
        block = summary["independent_fixtures"][key]
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 3.4, f"{rate:.1f}%", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color=INK)
        ax.text(bar.get_x() + bar.get_width() / 2, 4, f"{block['wins']}/{block['n']}", ha="center", va="bottom",
                fontsize=10, color=WHITE, fontweight="bold")
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_title("每关独立开局单场胜率", fontsize=12, pad=10)

    fig.subplots_adjust(wspace=0.3)
    save(fig, "clear-rate.png")


def per_opponent_chart(summary):
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    style(ax)
    height = 0.36
    ys = list(range(len(STAGES)))
    for offset, (key, label, color) in zip((-height / 2, height / 2), JITTERS):
        block = summary["natural_campaigns"][key]["per_opponent"]
        values = [block[enemy]["win_rate"] * 100 for enemy, _ in STAGES]
        counts = [f"{block[enemy]['wins']}/{block[enemy]['reached']}" for enemy, _ in STAGES]
        bars = ax.barh([y + offset for y in ys], values, height=height, color=color, zorder=3, label=label)
        for bar, value, count in zip(bars, values, counts):
            ax.text(value + 0.25, bar.get_y() + bar.get_height() / 2, count, va="center", fontsize=8.5,
                    color=MUTED)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{i + 1:02d} {zh}" for i, (_, zh) in enumerate(STAGES)], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(88, 103)
    ax.set_xlabel("胜率（到达该关的对局）")
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False, fontsize=10, ncols=2)
    ax.set_title("逐关胜率：将军与肯是主要失分点", fontsize=13, pad=12)
    save(fig, "per-opponent.png")


def wins_distribution_chart(runs):
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    style(ax)
    width = 0.38
    top = 1
    for offset, (key, label, color) in zip((-width / 2, width / 2), JITTERS):
        jitter = float(key)
        counts = Counter(run["wins"] for run in runs if abs(run["jitter"] - jitter) < 1e-9)
        xs = list(range(0, 13))
        ys = [counts.get(x, 0) for x in xs]
        top = max(top, max(ys))
        bars = ax.bar([x + offset for x in xs], ys, width=width, color=color, zorder=3, label=label)
        for bar, value in zip(bars, ys):
            if value:
                ax.text(bar.get_x() + bar.get_width() / 2, value * 1.12, str(value), ha="center", va="bottom",
                        fontsize=7.5, color=MUTED)
    ax.set_yscale("log")
    ax.set_ylim(0.8, top * 3.2)
    ax.set_xticks(range(0, 13))
    ax.set_xlabel("通关场数（12 场全胜才算一币通关）")
    ax.set_ylabel("次数（对数刻度）")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=10)
    ax.set_title("一币赛程的赢场分布", fontsize=13, pad=12)
    save(fig, "wins-distribution.png")


if __name__ == "__main__":
    summary, runs = load_results()
    clear_rate_chart(summary)
    per_opponent_chart(summary)
    wins_distribution_chart(runs)
