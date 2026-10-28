#!/usr/bin/env python3
"""
Volcanic Island (Chapter 7) — semantic mechanism declines, category persists
===========================================================================
This is a didactic mechanism sketch: an intuition pump, not evidence.

Core idea:
- Agents learn a binary category boundary on a continuous cue x ("shape"/"distributional cue")
- There is also a semantic cue s ∈ {+1,-1} that is informative pre-switch and uninformative post-switch
- We track:
    A) semantic cue effect in agent predictions and mean learned semantic weight w
    B) persistence of the category: boundary similarity to the pre-switch mean, and consensus (sharpness)

Deliberate simplifications:
- Agents learn from world-generated tokens, not direct pairwise dialogue.
- Persistence uses population-level summary metrics (mean boundary + consensus).
- Post-switch environment includes a mild x-distribution shift (x_mu_2) to avoid trivial stasis.
- This model is for mechanism plausibility, not historical calibration.
- If population size grows, interaction horizon must scale too; otherwise each
  agent receives too few updates for meaningful learning dynamics.
"""

import argparse, math, random, pathlib, shutil, json
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
HAS_MATPLOTLIB = True

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
CODE_DIR = pathlib.Path(__file__).resolve().parent
FIGURES_DIR = ROOT_DIR / "figures"


def _save_manifest(filename: str, payload: Dict):
    """
    Save a lightweight run manifest alongside generated figures.
    """
    payload = dict(payload)
    payload["generated_utc"] = datetime.now(timezone.utc).isoformat()

    targets = [
        FIGURES_DIR / filename,
        CODE_DIR / filename,
    ]
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    print(f"Manifest saved to: {FIGURES_DIR / filename}")

# -----------------------------
# House Style & Config
# -----------------------------

PALETTE = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "vermillion": "#D55E00", "purple": "#CC79A7", "gray": "#4D4D4D"
}

DEFAULT_N_AGENTS = 200
DEFAULT_T = 24000
DEFAULT_T_SWITCH = 12000
DEFAULT_UPDATES_PER_AGENT = DEFAULT_T / DEFAULT_N_AGENTS

def setup_plot_style():
    plt.rcParams.update({
        'font.family': 'serif', 'mathtext.fontset': 'cm',
        'axes.labelsize': 10, 'axes.titlesize': 11,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 9, 'figure.dpi': 300,
        'lines.linewidth': 1.6, 'axes.grid': True,
        'grid.linestyle': ':', 'grid.alpha': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False,
    })

# -----------------------------
# Model
# -----------------------------

@dataclass
class Agent:
    theta: float  # boundary on x
    w: float      # weight on semantic cue s
    lr: float = 0.02

    def predict_prob(self, x: float, s: int) -> float:
        z = (x - self.theta) + self.w * s
        # Logistic; clamp for numerical stability
        z = max(-20.0, min(20.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def update(
        self,
        x: float,
        s: int,
        y: int,
        freeze_theta: bool = False,
        freeze_w: bool = False
    ):
        """
        One-step SGD on logistic loss for parameters (theta, w).
        y ∈ {0,1}, s ∈ {+1,-1}.
        """
        p = self.predict_prob(x, s)
        err = (p - y)  # dL/dz
        # d z / d theta = -1 ; d z / d w = s
        if not freeze_theta:
            self.theta -= self.lr * (-err)
        if not freeze_w:
            self.w -= self.lr * (err * s)

def gen_token(t: int, t_switch: int,
              x_mu_1: float, x_mu_2: float, x_sigma: float,
              semantic_reliability_1: float, semantic_reliability_2: float) -> Tuple[float, int, int]:
    """
    Generate (x, s, y) from the "world".

    y is determined by x against a fixed true boundary at 0.0.
    s is correlated with y pre-switch and (optionally) becomes uncorrelated post-switch.

    semantic_reliability = P(s matches y-as-sign), where y-sign is +1 for y=1, -1 for y=0.
    """
    mu = x_mu_1 if t < t_switch else x_mu_2
    rel = semantic_reliability_1 if t < t_switch else semantic_reliability_2

    x = random.gauss(mu, x_sigma)
    y = 1 if x >= 0.0 else 0

    y_sign = +1 if y == 1 else -1
    if random.random() < rel:
        s = y_sign
    else:
        s = -y_sign
    return x, s, y

def simulate(n_agents: int = 200,
             T: int = 24000,
             t_switch: int = 12000,
             x_mu_1: float = 0.0,
             x_mu_2: float = 0.15,   # mild ecological shift after switch -> coherent boundary drift
             x_sigma: float = 1.0,
             semantic_reliability_1: float = 0.65,
             semantic_reliability_2: float = 0.50,
             lr: float = 0.02,
             window: int = 250,
             switch_semantic: bool = True,
             switch_ecology: bool = True,
             freeze_w_after_switch: bool = False,
             freeze_theta_after_switch: bool = False) -> Dict[str, np.ndarray]:
    """
    Returns a dict of time series.
    """
    agents: List[Agent] = []
    for _ in range(n_agents):
        agents.append(Agent(theta=random.gauss(0, 0.25), w=random.gauss(0, 0.05), lr=lr))

    # Track pre-switch reference boundary mean (estimated online)
    pre_switch_theta_means: List[float] = []

    # Rolling window for semantic cue effect in agent output
    buf_s: List[int] = []
    buf_pred: List[int] = []

    # Series
    times = []
    sem_effect = []
    w_mean = []
    theta_mean = []
    theta_std = []
    # Post-switch only: similarity to the pre-switch mean
    boundary_sim = []
    boundary_sim_t = []
    consensus = []

    for t in range(T):
        # Sample a random learner; world generates token
        a = random.choice(agents)
        x_mu_2_eff = x_mu_2 if switch_ecology else x_mu_1
        sem_rel_2_eff = semantic_reliability_2 if switch_semantic else semantic_reliability_1
        x, s, y = gen_token(
            t,
            t_switch,
            x_mu_1,
            x_mu_2_eff,
            x_sigma,
            semantic_reliability_1,
            sem_rel_2_eff
        )
        freeze_w = freeze_w_after_switch and (t >= t_switch)
        freeze_theta = freeze_theta_after_switch and (t >= t_switch)
        # Record the agent's current output before learning from this token.
        y_hat = 1 if a.predict_prob(x, s) >= 0.5 else 0
        a.update(x, s, y, freeze_theta=freeze_theta, freeze_w=freeze_w)

        # Collect reference stats
        th = np.array([ag.theta for ag in agents])
        ww = np.array([ag.w for ag in agents])

        if t < t_switch:
            pre_switch_theta_means.append(th.mean())

        # rolling semantic effect
        buf_s.append(s)
        buf_pred.append(y_hat)
        if len(buf_s) > window:
            buf_s.pop(0); buf_pred.pop(0)

        if (t % window) == 0 and t > 0:
            times.append(t)

            # semantic cue effect in model output:
            # P(pred=1|s=+) - P(pred=1|s=-) in the recent window.
            s_arr = np.array(buf_s)
            pred_arr = np.array(buf_pred)
            p_pos = pred_arr[s_arr == +1].mean() if np.any(s_arr == +1) else np.nan
            p_neg = pred_arr[s_arr == -1].mean() if np.any(s_arr == -1) else np.nan
            sem_effect.append(p_pos - p_neg)

            w_mean.append(ww.mean())
            theta_mean.append(th.mean())
            theta_std.append(th.std())

            # Consensus (sharpness): 1/(1+sd) so higher = tighter clustering.
            consensus.append(1.0 / (1.0 + float(th.std())))

            # Boundary similarity, only meaningful once we have a pre-switch baseline
            if t >= t_switch and len(pre_switch_theta_means) > 10:
                pre_mean = float(np.mean(pre_switch_theta_means))
                # Similarity = 1 - normalized distance
                sim = 1.0 - min(1.0, abs(float(th.mean()) - pre_mean) / 1.5)  # 1.5 is a scale, not "tuning"
                boundary_sim.append(sim)
                boundary_sim_t.append(t)

    out = {
        "times": np.array(times),
        "sem_effect": np.array(sem_effect),
        "w_mean": np.array(w_mean),
        "theta_mean": np.array(theta_mean),
        "theta_std": np.array(theta_std),
        "boundary_sim_t": np.array(boundary_sim_t),
        "boundary_sim": np.array(boundary_sim),
        "consensus": np.array(consensus),
        # diagnostics snapshots
        "w_final": np.array([ag.w for ag in agents]),
        "theta_final": np.array([ag.theta for ag in agents]),
        "pre_switch_theta_mean": float(np.mean(pre_switch_theta_means)) if pre_switch_theta_means else float("nan"),
    }
    return out

# -----------------------------
# Plotting and diagnostics
# -----------------------------

def make_main_plot(series: Dict[str, np.ndarray], t_switch: int, out_png: str):
    setup_plot_style()
    times = series["times"]
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5), sharex=False)

    # A: semantic mechanism
    ax[0].plot(times, series["sem_effect"], color=PALETTE["vermillion"], lw=1.6, label="Semantic cue effect")
    ax[0].axvline(t_switch, color=PALETTE["gray"], linestyle="--", alpha=0.6)
    ax[0].set_title("A. Decline of semantic mechanism", loc="left")
    ax[0].set_xlabel("Interactions (Time)")
    ax[0].set_ylabel("Cue effect")
    ax[0].legend(loc="lower left", frameon=False)

    # B: persistence
    ax[1].plot(series["boundary_sim_t"], series["boundary_sim"], color=PALETTE["green"], lw=1.6, label="Boundary sim. (to pre-switch)")
    ax[1].plot(times, series["consensus"], color=PALETTE["orange"], lw=1.6, label="Consensus (sharpness)")
    ax[1].axvline(t_switch, color=PALETTE["gray"], linestyle="--", alpha=0.6)
    ax[1].set_title("B. Persistence of the category", loc="left")
    ax[1].set_xlabel("Interactions (Time)")
    ax[1].set_ylabel("Stability / Agreement")
    ax[1].set_ylim(0.0, 1.05)
    ax[1].legend(loc="lower left", frameon=False)

    plt.tight_layout()
    plt.savefig(out_png)
    out_pdf = out_png.replace('.png', '.pdf')
    plt.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved: {out_png}\nSaved: {out_pdf}")

def diagnose(series: Dict[str, np.ndarray], out_png: str):
    w = series["w_final"]
    th = series["theta_final"]

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), dpi=180)

    ax[0].hist(w, bins=30)
    ax[0].set_title("Final w distribution")
    ax[0].set_xlabel("w")
    ax[0].set_ylabel("count")

    ax[1].bar(["mean w", "mean |w|"], [float(np.mean(w)), float(np.mean(np.abs(w)))])
    ax[1].set_title("Mean vs mean |w|")

    ax[2].hist(th, bins=30)
    ax[2].axvline(series["pre_switch_theta_mean"], linestyle="--")
    ax[2].set_title("Final boundary θ distribution")
    ax[2].set_xlabel("θ")

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)
    print(f"Saved diagnostics: {out_png}")


def _sync_plot_outputs(base_name: str):
    """Copy a code-side plot (png/pdf) into figures/ when present."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        src = CODE_DIR / f"{base_name}{ext}"
        dst = FIGURES_DIR / f"{base_name}{ext}"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied: {dst}")


def _series_summary(series: Dict[str, np.ndarray], t_switch: int = 12000) -> Dict[str, float]:
    """Compute compact pre/post summary stats for reporting and manifests."""
    times = series["times"]
    pre_mask = times < t_switch
    post_mask = times >= t_switch
    tail = 5

    def _tail_mean(values: np.ndarray, mask: np.ndarray) -> float:
        subset = np.asarray(values[mask], dtype=float)
        if subset.size == 0:
            return float("nan")
        return float(np.nanmean(subset[-tail:]))

    return {
        "semantic_effect_pre_mean": float(np.nanmean(series["sem_effect"][pre_mask])) if np.any(pre_mask) else float("nan"),
        "semantic_effect_post_mean": float(np.nanmean(series["sem_effect"][post_mask])) if np.any(post_mask) else float("nan"),
        "semantic_effect_pre_late_mean": _tail_mean(series["sem_effect"], pre_mask),
        "semantic_effect_post_late_mean": _tail_mean(series["sem_effect"], post_mask),
        "w_pre_mean": float(np.nanmean(series["w_mean"][pre_mask])) if np.any(pre_mask) else float("nan"),
        "w_post_mean": float(np.nanmean(series["w_mean"][post_mask])) if np.any(post_mask) else float("nan"),
        "w_pre_late_mean": _tail_mean(series["w_mean"], pre_mask),
        "w_post_late_mean": _tail_mean(series["w_mean"], post_mask),
        "w_final": float(series["w_mean"][-1]) if len(series["w_mean"]) else float("nan"),
        "boundary_similarity_post_mean": float(np.nanmean(series["boundary_sim"])) if len(series["boundary_sim"]) else float("nan"),
        "consensus_post_mean": float(np.nanmean(series["consensus"][post_mask])) if np.any(post_mask) else float("nan"),
        "consensus_post_late_mean": _tail_mean(series["consensus"], post_mask),
    }


def _parse_reliability_list(spec: str) -> List[float]:
    values: List[float] = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        v = float(raw)
        if not (0.0 < v < 1.0):
            raise ValueError(f"Reliability must be in (0,1), got {v}")
        values.append(v)
    if not values:
        raise ValueError("Reliability list is empty.")
    return values


def _resolve_schedule(
    n_agents: int,
    T: Optional[int],
    t_switch: Optional[int],
    updates_per_agent_target: Optional[float],
    t_switch_fraction: float = 0.5,
) -> Tuple[int, int, float]:
    if n_agents <= 0:
        raise ValueError("n_agents must be > 0")
    if t_switch_fraction <= 0.0 or t_switch_fraction >= 1.0:
        raise ValueError("t_switch_fraction must be in (0,1)")

    if T is None:
        target = (
            updates_per_agent_target
            if updates_per_agent_target is not None
            else DEFAULT_UPDATES_PER_AGENT
        )
        if target <= 0:
            raise ValueError("updates_per_agent_target must be > 0")
        T_eff = max(2, int(round(target * n_agents)))
    else:
        T_eff = int(T)
        if T_eff <= 1:
            raise ValueError("T must be > 1")

    if t_switch is None:
        t_switch_eff = int(round(T_eff * t_switch_fraction))
    else:
        t_switch_eff = int(t_switch)

    t_switch_eff = min(max(1, t_switch_eff), T_eff - 1)
    return T_eff, t_switch_eff, T_eff / float(n_agents)


def _run_control_suite(
    seed: int,
    n_agents: int,
    T: int,
    t_switch: int = 12000,
    window: int = 250,
) -> Dict[str, Dict[str, float]]:
    """
    Run mechanism controls:
    - baseline
    - no semantic/ecological switch
    - freeze-w after switch
    - freeze-theta after switch
    """
    scenarios = [
        ("baseline", {}),
        ("no_switch", {"switch_semantic": False, "switch_ecology": False}),
        ("freeze_w", {"freeze_w_after_switch": True}),
        ("freeze_theta", {"freeze_theta_after_switch": True}),
    ]

    summaries: Dict[str, Dict[str, float]] = {}
    for name, kwargs in scenarios:
        random.seed(seed)
        np.random.seed(seed)
        series = simulate(
            n_agents=n_agents,
            T=T,
            t_switch=t_switch,
            window=window,
            **kwargs,
        )
        summaries[name] = _series_summary(series, t_switch=t_switch)

    print("\n" + "=" * 70)
    print("VOLCANO CONTROL SUITE")
    print("=" * 70)
    print("Scenario       sem_pre   sem_post   w_pre   w_post   bsim_post   cons_post")
    for name in ["baseline", "no_switch", "freeze_w", "freeze_theta"]:
        s = summaries[name]
        print(
            f"{name:<12} "
            f"{s['semantic_effect_pre_mean']:+.3f}   "
            f"{s['semantic_effect_post_mean']:+.3f}   "
            f"{s['w_pre_mean']:+.3f}   "
            f"{s['w_post_mean']:+.3f}   "
            f"{s['boundary_similarity_post_mean']:+.3f}   "
            f"{s['consensus_post_mean']:+.3f}"
        )

    if HAS_MATPLOTLIB:
        setup_plot_style()
        labels = ["baseline", "no_switch", "freeze_w", "freeze_theta"]
        x = np.arange(len(labels))
        sem_post = [summaries[k]["semantic_effect_post_mean"] for k in labels]
        bsim_post = [summaries[k]["boundary_similarity_post_mean"] for k in labels]
        cons_post = [summaries[k]["consensus_post_mean"] for k in labels]

        fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=180)
        ax[0].bar(x, sem_post, color=PALETTE["vermillion"])
        ax[0].axhline(0.0, color=PALETTE["gray"], linewidth=1, linestyle=":")
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(labels, rotation=20, ha="right")
        ax[0].set_title("Post-switch semantic effect", loc="left")
        ax[0].set_ylabel("Mean semantic cue effect")

        width = 0.35
        ax[1].bar(x - width / 2, bsim_post, width=width, color=PALETTE["green"], label="Boundary sim.")
        ax[1].bar(x + width / 2, cons_post, width=width, color=PALETTE["orange"], label="Consensus")
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(labels, rotation=20, ha="right")
        ax[1].set_ylim(0.0, 1.05)
        ax[1].set_title("Post-switch persistence", loc="left")
        ax[1].legend(loc="lower left", frameon=False)
        plt.tight_layout()

        out_png = CODE_DIR / "volcano_control_experiments.png"
        out_pdf = CODE_DIR / "volcano_control_experiments.pdf"
        plt.savefig(out_png)
        plt.savefig(out_pdf)
        plt.close(fig)
        _sync_plot_outputs("volcano_control_experiments")

    _save_manifest(
        "volcano_control_experiments_manifest.json",
        {
            "experiment": "volcano_controls",
            "seed": seed,
            "n_agents": n_agents,
            "T": T,
            "t_switch": t_switch,
            "window": window,
            "updates_per_agent": T / float(n_agents),
            "summaries": summaries,
        },
    )
    return summaries


def _run_reliability_sweep(
    seed: int,
    n_agents: int,
    T: int,
    t_switch: int = 12000,
    window: int = 250,
    reliabilities: Optional[List[float]] = None
) -> List[Dict[str, float]]:
    """Sweep post-switch semantic reliability and track outcome sensitivity."""
    if reliabilities is None:
        reliabilities = [0.50, 0.55, 0.60, 0.65]

    rows: List[Dict[str, float]] = []
    for rel in reliabilities:
        random.seed(seed)
        np.random.seed(seed)
        series = simulate(
            n_agents=n_agents,
            T=T,
            t_switch=t_switch,
            window=window,
            semantic_reliability_2=rel,
        )
        summary = _series_summary(series, t_switch=t_switch)
        row = {"semantic_reliability_2": rel, **summary}
        rows.append(row)

    print("\n" + "=" * 70)
    print("VOLCANO RELIABILITY SWEEP")
    print("=" * 70)
    print("rel2   sem_post   w_post   bsim_post   cons_post")
    for r in rows:
        print(
            f"{r['semantic_reliability_2']:.2f}   "
            f"{r['semantic_effect_post_mean']:+.3f}   "
            f"{r['w_post_mean']:+.3f}   "
            f"{r['boundary_similarity_post_mean']:+.3f}   "
            f"{r['consensus_post_mean']:+.3f}"
        )

    if HAS_MATPLOTLIB:
        setup_plot_style()
        x = [r["semantic_reliability_2"] for r in rows]
        sem_post = [r["semantic_effect_post_mean"] for r in rows]
        bsim_post = [r["boundary_similarity_post_mean"] for r in rows]
        cons_post = [r["consensus_post_mean"] for r in rows]

        fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=180)
        ax[0].plot(x, sem_post, marker="o", color=PALETTE["vermillion"])
        ax[0].axhline(0.0, color=PALETTE["gray"], linewidth=1, linestyle=":")
        ax[0].set_xlabel("Post-switch semantic reliability")
        ax[0].set_ylabel("Post-switch semantic effect")
        ax[0].set_title("Semantic effect sensitivity", loc="left")

        ax[1].plot(x, bsim_post, marker="o", color=PALETTE["green"], label="Boundary sim.")
        ax[1].plot(x, cons_post, marker="s", color=PALETTE["orange"], label="Consensus")
        ax[1].set_xlabel("Post-switch semantic reliability")
        ax[1].set_ylim(0.0, 1.05)
        ax[1].set_title("Persistence sensitivity", loc="left")
        ax[1].legend(loc="lower left", frameon=False)
        plt.tight_layout()

        out_png = CODE_DIR / "volcano_reliability_sweep.png"
        out_pdf = CODE_DIR / "volcano_reliability_sweep.pdf"
        plt.savefig(out_png)
        plt.savefig(out_pdf)
        plt.close(fig)
        _sync_plot_outputs("volcano_reliability_sweep")

    _save_manifest(
        "volcano_reliability_sweep_manifest.json",
        {
            "experiment": "volcano_reliability_sweep",
            "seed": seed,
            "n_agents": n_agents,
            "T": T,
            "t_switch": t_switch,
            "window": window,
            "updates_per_agent": T / float(n_agents),
            "rows": rows,
        },
    )
    return rows


def _run_seed_sensitivity(
    seed_start: int,
    seed_end: int,
    n_agents: int,
    T: int,
    t_switch: int,
    window: int,
) -> List[Dict[str, float]]:
    if seed_end < seed_start:
        raise ValueError("seed_end must be >= seed_start")

    rows: List[Dict[str, float]] = []
    for seed in range(seed_start, seed_end + 1):
        random.seed(seed)
        np.random.seed(seed)
        series = simulate(
            n_agents=n_agents,
            T=T,
            t_switch=t_switch,
            window=window,
        )
        summary = _series_summary(series, t_switch=t_switch)
        rows.append({"seed": float(seed), **summary})

    sem_post = np.array([r["semantic_effect_post_mean"] for r in rows], dtype=float)
    bsim_post = np.array([r["boundary_similarity_post_mean"] for r in rows], dtype=float)
    cons_post = np.array([r["consensus_post_mean"] for r in rows], dtype=float)

    print("\n" + "=" * 70)
    print("VOLCANO SEED SENSITIVITY")
    print("=" * 70)
    print(
        f"Seeds {seed_start}..{seed_end} (n={len(rows)}), "
        f"N={n_agents}, T={T}, updates/agent={T/float(n_agents):.1f}"
    )
    print(
        "Post-switch means ± sd: "
        f"semantic={np.nanmean(sem_post):+.3f}±{np.nanstd(sem_post):.3f}, "
        f"boundary_sim={np.nanmean(bsim_post):+.3f}±{np.nanstd(bsim_post):.3f}, "
        f"consensus={np.nanmean(cons_post):+.3f}±{np.nanstd(cons_post):.3f}"
    )

    if HAS_MATPLOTLIB:
        setup_plot_style()
        seeds = [int(r["seed"]) for r in rows]
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), dpi=180, sharex=True)

        ax[0].plot(seeds, sem_post, marker="o", color=PALETTE["vermillion"])
        ax[0].axhline(float(np.nanmean(sem_post)), color=PALETTE["gray"], linestyle=":")
        ax[0].set_title("Post-switch semantic effect", loc="left")
        ax[0].set_ylabel("mean")

        ax[1].plot(seeds, bsim_post, marker="o", color=PALETTE["green"])
        ax[1].axhline(float(np.nanmean(bsim_post)), color=PALETTE["gray"], linestyle=":")
        ax[1].set_title("Post-switch boundary similarity", loc="left")
        ax[1].set_ylim(0.0, 1.05)

        ax[2].plot(seeds, cons_post, marker="o", color=PALETTE["orange"])
        ax[2].axhline(float(np.nanmean(cons_post)), color=PALETTE["gray"], linestyle=":")
        ax[2].set_title("Post-switch consensus", loc="left")
        ax[2].set_ylim(0.0, 1.05)

        for axis in ax:
            axis.set_xlabel("Seed")

        plt.tight_layout()
        out_png = CODE_DIR / "volcano_seed_sensitivity.png"
        out_pdf = CODE_DIR / "volcano_seed_sensitivity.pdf"
        plt.savefig(out_png)
        plt.savefig(out_pdf)
        plt.close(fig)
        _sync_plot_outputs("volcano_seed_sensitivity")

    _save_manifest(
        "volcano_seed_sensitivity_manifest.json",
        {
            "experiment": "volcano_seed_sensitivity",
            "seed_start": seed_start,
            "seed_end": seed_end,
            "n_agents": n_agents,
            "T": T,
            "t_switch": t_switch,
            "window": window,
            "updates_per_agent": T / float(n_agents),
            "rows": rows,
            "aggregate": {
                "semantic_effect_post_mean_mean": float(np.nanmean(sem_post)),
                "semantic_effect_post_mean_sd": float(np.nanstd(sem_post)),
                "boundary_similarity_post_mean_mean": float(np.nanmean(bsim_post)),
                "boundary_similarity_post_mean_sd": float(np.nanstd(bsim_post)),
                "consensus_post_mean_mean": float(np.nanmean(cons_post)),
                "consensus_post_mean_sd": float(np.nanstd(cons_post)),
            },
        },
    )
    return rows

def main():
    ap = argparse.ArgumentParser(
        description="Generate Chapter 7 volcanic-island ABM plots."
    )
    ap.add_argument(
        "--exp",
        default="main",
        choices=["main", "controls", "sweep", "sensitivity", "all"],
        help="Which experiment set to run",
    )
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")
    ap.add_argument("--seed-start", type=int, default=1,
                    help="Start seed for --exp=sensitivity")
    ap.add_argument("--seed-end", type=int, default=20,
                    help="End seed for --exp=sensitivity")
    ap.add_argument("--n-agents", type=int, default=DEFAULT_N_AGENTS,
                    help="Population size")
    ap.add_argument("--T", type=int, default=None,
                    help="Total interactions (if omitted, derived from --updates-per-agent-target)")
    ap.add_argument("--t-switch", type=int, default=None,
                    help="Switch timestep (if omitted, derived from --t-switch-frac)")
    ap.add_argument("--t-switch-frac", type=float, default=0.5,
                    help="Switch fraction of T when --t-switch is omitted")
    ap.add_argument("--updates-per-agent-target", type=float, default=DEFAULT_UPDATES_PER_AGENT,
                    help="Used only when --T is omitted; keeps per-agent exposure comparable across N")
    ap.add_argument("--window", type=int, default=250,
                    help="Rolling diagnostic window")
    ap.add_argument("--reliabilities", default="0.50,0.55,0.60,0.65",
                    help="Comma-separated post-switch semantic reliabilities for --exp=sweep")
    args = ap.parse_args()

    T_eff, t_switch_eff, updates_per_agent = _resolve_schedule(
        n_agents=args.n_agents,
        T=args.T,
        t_switch=args.t_switch,
        updates_per_agent_target=args.updates_per_agent_target,
        t_switch_fraction=args.t_switch_frac,
    )
    reliabilities = _parse_reliability_list(args.reliabilities)

    print("\n" + "=" * 70)
    print("VOLCANO ABM RUN CONFIG")
    print("=" * 70)
    print(
        f"exp={args.exp}, seed={args.seed}, n_agents={args.n_agents}, "
        f"T={T_eff}, t_switch={t_switch_eff}, updates/agent={updates_per_agent:.1f}, "
        f"window={args.window}"
    )
    if updates_per_agent < 10.0:
        print("WARNING: updates/agent is very low; trajectories will be under-trained.")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if args.exp in ("main", "all"):
        random.seed(args.seed)
        np.random.seed(args.seed)
        series = simulate(
            n_agents=args.n_agents,
            T=T_eff,
            t_switch=t_switch_eff,
            window=args.window,
        )
        out_png = str(CODE_DIR / "volcano_drift_experiment.png")
        make_main_plot(series, t_switch=t_switch_eff, out_png=out_png)

        # Keep chapter-facing figures synchronized with code-side diagnostics.
        for ext in (".png", ".pdf"):
            src = CODE_DIR / f"volcano_drift_experiment{ext}"
            dst = FIGURES_DIR / f"volcano_drift_experiment{ext}"
            shutil.copy2(src, dst)
            print(f"Copied: {dst}")

        _save_manifest(
            "volcano_drift_experiment_manifest.json",
            {
                "experiment": "volcano_drift",
                "seed": args.seed,
                "parameters": {
                    "n_agents": args.n_agents,
                    "T": T_eff,
                    "t_switch": t_switch_eff,
                    "updates_per_agent": updates_per_agent,
                    "x_mu_1": 0.0,
                    "x_mu_2": 0.15,
                    "x_sigma": 1.0,
                    "semantic_reliability_1": 0.65,
                    "semantic_reliability_2": 0.50,
                    "lr": 0.02,
                    "window": args.window,
                    "boundary_similarity_scale": 1.5,
                },
                "summary": _series_summary(series, t_switch=t_switch_eff),
            },
        )

    if args.exp in ("controls", "all"):
        _run_control_suite(
            seed=args.seed,
            n_agents=args.n_agents,
            T=T_eff,
            t_switch=t_switch_eff,
            window=args.window,
        )

    if args.exp in ("sweep", "all"):
        _run_reliability_sweep(
            seed=args.seed,
            n_agents=args.n_agents,
            T=T_eff,
            t_switch=t_switch_eff,
            window=args.window,
            reliabilities=reliabilities,
        )

    if args.exp in ("sensitivity", "all"):
        _run_seed_sensitivity(
            seed_start=args.seed_start,
            seed_end=args.seed_end,
            n_agents=args.n_agents,
            T=T_eff,
            t_switch=t_switch_eff,
            window=args.window,
        )

if __name__ == "__main__":
    main()
