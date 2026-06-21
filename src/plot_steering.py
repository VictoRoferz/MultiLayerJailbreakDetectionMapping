#!/usr/bin/env python3
"""
plot_steering.py — figures from steering_sanity.py output.

Figure 1 (dose-response): ASR vs steering strength c, one panel per arm
(refused-start / benign-start), one line per steer config (single vs multi),
with the c=0 baseline included and the random-direction control overlaid if
present. This is the headline figure: shows multi-layer >> single, refused >>
benign, the peak, and the off-manifold collapse.

Usage (repo root):
    python src/plot_steering.py
    python src/plot_steering.py --json artifacts/steering_sanity.json --out artifacts/paper/figures
"""

import argparse
import json
from pathlib import Path


def _curve(entries, baseline_asr):
    """Return (xs, ys) sorted by c, with the c=0 baseline prepended."""
    pts = [(0.0, baseline_asr)] + [(e["c"], e["asr"]) for e in entries]
    pts = sorted(set(pts))
    return [p[0] for p in pts], [p[1] * 100 for p in pts]


def _random_curve(entries, baseline_asr):
    pts = [(0.0, baseline_asr * 100)]
    for e in entries:
        if "random_asr" in e and e["random_asr"] is not None:
            pts.append((e["c"], e["random_asr"] * 100))
    if len(pts) <= 1:
        return None, None
    pts = sorted(set(pts))
    return [p[0] for p in pts], [p[1] for p in pts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="artifacts/steering_sanity.json")
    ap.add_argument("--out", default="artifacts/paper/figures")
    args = ap.parse_args()

    data = json.loads(Path(args.json).read_text())
    arms = data.get("arms", {})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arm_names = [a for a in ("refused", "benign") if a in arms]
    fig, axes = plt.subplots(1, len(arm_names), figsize=(6 * len(arm_names), 4.5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    colors = plt.cm.tab10.colors
    for ax, arm in zip(axes, arm_names):
        a = arms[arm]
        base = a.get("baseline", {}).get("asr", 0.0)
        cfg_names = [k for k in a if isinstance(a[k], list)]
        for i, cfg in enumerate(sorted(cfg_names)):
            xs, ys = _curve(a[cfg], base)
            ax.plot(xs, ys, "o-", color=colors[i % 10], label=cfg)
            rxs, rys = _random_curve(a[cfg], base)
            if rxs is not None:
                ax.plot(rxs, rys, "x--", color=colors[i % 10], alpha=0.5,
                        label=f"{cfg} (random)")
        ax.axhline(base * 100, color="grey", ls=":", lw=1, label="baseline (c=0)")
        ax.set_title(f"{arm}-start")
        ax.set_xlabel("steering strength c")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Attack success rate (%)")
    fig.suptitle("Rank-1 refusal-axis steering: dose-response")
    fig.tight_layout()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "steering_dose_response.png"
    fig.savefig(path, dpi=150)
    print(f"[done] wrote {path}")


if __name__ == "__main__":
    main()
