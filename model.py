"""
Unbiased Top 100 — PyMC model
=================================
Corrects for alphabetical bias in DR's Top 100 voting.

WHY THIS MODEL
--------------
An earlier version used a hierarchical Poisson model with 100 per-song latent
quality parameters (α_i) plus a bias coefficient β.  That model is
fundamentally under-identified: with 100 latent parameters for 100
observations, β is only constrained by the hierarchical prior, not by genuine
data signal.  The result — β = −0.045 ± 0.082, a credible interval that
easily spans zero — reflects this lack of identification, not a genuine
absence of bias.

The alphabetical bias also primarily operates at SELECTION level: it
determines which songs make it into the top 100.  Within the top 100, the
rank-level signal is necessarily weaker because all songs already cleared
the bias-inflated threshold.

REVISED MODEL
-------------
Simple Bayesian linear regression with no per-song effects:

  α      ~ Normal(log(median_votes), 1)   # global intercept
  β      ~ Normal(0, 0.5)                 # alphabetical bias slope
  σ      ~ HalfNormal(1)                  # residual spread (quality + noise)
  μ_i     = α + β × pos_std_i
  log(votes_i) ~ Normal(μ_i, σ)

β is now the slope of a simple regression line — directly identifiable
from the data.  σ honestly absorbs all unexplained variation, including
genuine song-quality differences.

INTERPRETATION
--------------
β < 0  →  earlier-alphabet artists tend to accumulate more votes on average.
The corrected ranking subtracts the estimated alphabetical contribution from
each song's log-votes and re-ranks.

LIMITATION
----------
This measures the CORRELATION between alphabetical position and rank.  It
conflates genuine quality differences with voting bias.  Without the full
ballot data (all ~200–300 songs that were on offer) we cannot fully separate
the two effects.

Vote proxy: votes_i = 101 − rank_i  (rank 1 → 100, rank 100 → 1)
Alphabetical position: first letter of the artist's first name token,
  mapped to 1..29 using the Danish alphabet (A=1 … Ø=28 … Å=29).
"""

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

# ---------------------------------------------------------------------------
# Danish alphabet — 29 letters (A–Z plus Æ, Ø, Å)
# ---------------------------------------------------------------------------
DANISH_ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["Æ", "Ø", "Å"]
LETTER_RANK = {letter: i + 1 for i, letter in enumerate(DANISH_ALPHABET)}


def danish_sort_key(artist_raw: str) -> int:
    """
    Return 1-based alphabetical position (1=A … 29=Å) for the first
    meaningful token of the artist string.

    Rules:
    - Strip leading 'Tekst:' prefix (e.g. 'Tekst: B. S. Ingemann' → 'B')
    - Take the first whitespace-delimited token
    - Use its first Unicode letter character
    - Normalise to NFC, uppercase, map to Danish-alphabet position
    - Fall back to position 14 (M, middle of alphabet) if unmappable
    """
    # Strip "Tekst:" prefix
    cleaned = re.sub(r"(?i)^tekst:\s*", "", artist_raw.strip())
    # Take first token
    first_token = cleaned.split()[0] if cleaned.split() else cleaned
    # Get first letter character
    for ch in first_token:
        if unicodedata.category(ch).startswith("L"):
            letter = unicodedata.normalize("NFC", ch).upper()
            # Map common composed Danish letters
            letter = letter.replace("\u00c6", "Æ").replace("\u00d8", "Ø").replace("\u00c5", "Å")
            pos = LETTER_RANK.get(letter)
            if pos is not None:
                return pos
    return 14  # fallback to middle


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data(path: str = "top100.txt") -> pd.DataFrame:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        rank_str, title, artist = parts[0], parts[1], parts[2]
        try:
            rank = int(rank_str)
        except ValueError:
            continue
        rows.append({"rank_biased": rank, "song_title": title, "artist": artist})

    df = pd.DataFrame(rows)
    df["votes"] = 101 - df["rank_biased"]
    df["alpha_pos"] = df["artist"].apply(danish_sort_key)
    df["alpha_letter"] = df["artist"].apply(
        lambda a: re.sub(r"(?i)^tekst:\s*", "", a.strip()).split()[0][0].upper()
        if a.strip() else "?"
    )
    df = df.sort_values("rank_biased").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# PyMC model — simple Bayesian linear regression
# ---------------------------------------------------------------------------
def build_and_sample(df: pd.DataFrame):
    votes_obs = df["votes"].values.astype(float)
    alpha_pos = df["alpha_pos"].values.astype(float)

    # Standardise alphabetical position
    pos_mean = alpha_pos.mean()
    pos_std  = alpha_pos.std() if alpha_pos.std() > 0 else 1.0
    alpha_pos_std = (alpha_pos - pos_mean) / pos_std

    log_votes = np.log(votes_obs)

    with pm.Model() as model:
        # Global intercept (centred on observed log-vote mean)
        alpha = pm.Normal("alpha", mu=float(np.mean(log_votes)), sigma=1.0)

        # Alphabetical bias slope
        # Negative β ⇒ earlier-alphabet artists have higher (log) votes
        beta = pm.Normal("beta", mu=0.0, sigma=0.5)

        # Residual spread — captures all unexplained variation (quality + noise)
        sigma = pm.HalfNormal("sigma", sigma=1.0)

        # Linear predictor on log scale
        mu = alpha + beta * alpha_pos_std

        # Normal likelihood on log(votes)
        pm.Normal("log_votes_obs", mu=mu, sigma=sigma, observed=log_votes)

        print("Sampling…")
        idata = pm.sample(
            draws=2000,
            tune=2000,
            chains=4,
            target_accept=0.90,
            progressbar=True,
            random_seed=42,
        )

    return model, idata, pos_mean, pos_std


# ---------------------------------------------------------------------------
# Post-processing & output
# ---------------------------------------------------------------------------
def build_results(df: pd.DataFrame, idata, pos_mean: float, pos_std: float):
    from scipy import stats

    posterior = idata.posterior
    votes_obs = df["votes"].values.astype(float)
    alpha_pos = df["alpha_pos"].values.astype(float)
    alpha_pos_std = (alpha_pos - pos_mean) / pos_std

    beta_samples = posterior["beta"].values.flatten()           # all MCMC draws
    beta_mean    = float(beta_samples.mean())
    beta_sd      = float(beta_samples.std())
    beta_hdi     = az.hdi(idata, var_names=["beta"], prob=0.94)["beta"].values
    beta_raw     = beta_mean / pos_std                          # slope on raw 1..29 scale

    # Probability that β < 0 (bias in expected direction)
    prob_negative = float((beta_samples < 0).mean())

    print(f"\n=== Alphabetical bias coefficient ===")
    print(f"  β (standardised) = {beta_mean:.4f} ± {beta_sd:.4f}")
    print(f"  94% HDI          = [{beta_hdi[0]:.4f}, {beta_hdi[1]:.4f}]")
    print(f"  P(β < 0)         = {prob_negative:.3f}")
    print(f"  β (raw scale)    = {beta_raw:.5f}  log(votes) per letter")

    # ── Corrected ranking ────────────────────────────────────────────────
    # Remove estimated alphabetical contribution from each song's log-votes.
    # corrected_log_votes_i = log(votes_i) − β_mean × pos_std_i
    alphabetical_contribution = beta_mean * alpha_pos_std
    corrected_log_votes = np.log(votes_obs) - alphabetical_contribution
    df["corrected_log_votes"] = corrected_log_votes

    df["rank_unbiased"] = (
        df["corrected_log_votes"].rank(ascending=False, method="first").astype(int)
    )
    df["rank_delta"] = df["rank_biased"] - df["rank_unbiased"]  # positive = moved up

    # Bias effect per song in log-vote units (positive = alphabet gave a boost)
    # For early-alphabet songs: pos_std < 0, β < 0 → contribution > 0 (boosted)
    df["bias_effect"] = alphabetical_contribution  # same sign convention as above

    # ── Classical check: Spearman correlation (alpha_pos vs rank) ─────────
    rho, pval = stats.spearmanr(df["alpha_pos"], df["rank_biased"])
    print(f"\n=== Spearman ρ (alpha_pos vs rank_biased) ===")
    print(f"  ρ = {rho:.4f},  p = {pval:.4f}")
    print(f"  (positive ρ ⇒ earlier alphabet → lower rank number → better position)")

    # ── Build output records ──────────────────────────────────────────────
    records = []
    for _, row in df.sort_values("rank_unbiased").iterrows():
        records.append({
            "rank_biased":      int(row["rank_biased"]),
            "rank_unbiased":    int(row["rank_unbiased"]),
            "rank_delta":       int(row["rank_delta"]),
            "song_title":       row["song_title"],
            "artist":           row["artist"],
            "alpha_pos":        int(row["alpha_pos"]),
            "alpha_letter":     row["alpha_letter"],
            "bias_effect":      round(float(row["bias_effect"]), 4),
            # kept as "posterior_quality" for frontend compatibility
            "posterior_quality": round(float(row["corrected_log_votes"]), 4),
        })

    summary = {
        "beta_mean":      round(beta_mean,      4),
        "beta_sd":        round(beta_sd,         4),
        "beta_hdi_low":   round(float(beta_hdi[0]), 4),
        "beta_hdi_high":  round(float(beta_hdi[1]), 4),
        "beta_raw":       round(beta_raw,        5),
        "prob_negative":  round(prob_negative,   3),
        "spearman_rho":   round(float(rho),      4),
        "spearman_p":     round(float(pval),     4),
        "pos_mean":       round(pos_mean,        4),
        "pos_std":        round(pos_std,         4),
    }
    return records, summary


def save_results(records: list[dict], summary: dict, out_path: str = "docs/results.json"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "songs": records}
    Path(out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(records)} songs → {out_path}")


# ---------------------------------------------------------------------------
# Arviz diagnostic plots → docs/plots/
# ---------------------------------------------------------------------------
def save_plots(idata, out_dir: str = "docs/plots"):
    """
    Save four diagnostic PNGs using matplotlib + raw posterior samples.
    Avoids arviz_plots (1.x API is incompatible with older-style kwargs).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    var_names = ["alpha", "beta", "sigma"]
    labels    = {"alpha": "α (intercept)", "beta": "β (bias slope)", "sigma": "σ (residual)"}

    # ── helper: extract (chains, draws) array ───────────────────────────
    def get_samples(var):
        arr = idata.posterior[var].values  # (chain, draw)
        return arr  # shape (n_chains, n_draws)

    n_chains = idata.posterior.sizes["chain"]
    chain_colors = ["#6366f1", "#e63946", "#16a34a", "#f59e0b"][:n_chains]

    # ── 1. Posterior distributions ───────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    fig.suptitle("Posterior distributions (94% HDI)", fontsize=12)
    for ax, var in zip(axes, var_names):
        samples = get_samples(var).flatten()
        lo, hi = np.percentile(samples, [3, 97])
        xs = np.linspace(samples.min(), samples.max(), 400)
        kde = gaussian_kde(samples)
        ys = kde(xs)
        ax.plot(xs, ys, color="#6366f1", lw=2)
        mask = (xs >= lo) & (xs <= hi)
        ax.fill_between(xs[mask], ys[mask], alpha=0.25, color="#6366f1", label=f"94% HDI")
        ax.axvline(samples.mean(), color="#e63946", lw=1.5, ls="--", label=f"mean={samples.mean():.3f}")
        ax.set_title(labels[var], fontsize=10)
        ax.set_xlabel("value"); ax.set_ylabel("density")
        ax.legend(fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/posterior.png", dpi=130, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved {out_dir}/posterior.png")

    # ── 2. Trace plot ────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(11, 7))
    fig.suptitle("Trace plot — MCMC chains", fontsize=12)
    for row, var in enumerate(var_names):
        arr = get_samples(var)   # (chains, draws)
        ax_kde, ax_trace = axes[row]
        for c in range(n_chains):
            chain_samples = arr[c]
            # KDE
            xs = np.linspace(chain_samples.min(), chain_samples.max(), 300)
            kde = gaussian_kde(chain_samples)
            ax_kde.plot(xs, kde(xs), color=chain_colors[c], alpha=0.8, lw=1.5, label=f"chain {c}")
            # Trace
            ax_trace.plot(chain_samples, color=chain_colors[c], alpha=0.6, lw=0.5)
        ax_kde.set_ylabel(labels[var], fontsize=9)
        ax_kde.spines[["top","right"]].set_visible(False)
        ax_trace.spines[["top","right"]].set_visible(False)
        if row == 0:
            ax_kde.set_title("KDE per chain"); ax_trace.set_title("Samples over iterations")
    axes[-1][0].set_xlabel("density"); axes[-1][1].set_xlabel("iteration")
    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=n_chains, fontsize=9, frameon=False)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(f"{out_dir}/trace.png", dpi=130, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved {out_dir}/trace.png")

    # ── 3. Pair plot ─────────────────────────────────────────────────────
    flat = {v: get_samples(v).flatten() for v in var_names}
    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    fig.suptitle("Joint posterior (pair plot)", fontsize=12)
    for i, vi in enumerate(var_names):
        for j, vj in enumerate(var_names):
            ax = axes[i][j]
            if i == j:
                xs = np.linspace(flat[vi].min(), flat[vi].max(), 300)
                ax.plot(xs, gaussian_kde(flat[vi])(xs), color="#6366f1", lw=2)
                ax.set_yticks([])
            elif i > j:
                ax.scatter(flat[vj], flat[vi], s=1, alpha=0.15, color="#6366f1", rasterized=True)
            else:
                ax.set_visible(False)
            if j == 0: ax.set_ylabel(labels[vi], fontsize=8)
            if i == 2: ax.set_xlabel(labels[vj], fontsize=8)
            ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/pair.png", dpi=130, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved {out_dir}/pair.png")

    # ── 4. Autocorrelation ───────────────────────────────────────────────
    max_lag = 40
    fig, axes = plt.subplots(n_chains, 3, figsize=(11, 2.2 * n_chains))
    fig.suptitle("Autocorrelation (chain mixing)", fontsize=12)
    for c in range(n_chains):
        for j, var in enumerate(var_names):
            ax = axes[c][j] if n_chains > 1 else axes[j]
            s = get_samples(var)[c]
            s = s - s.mean()
            lags = range(max_lag + 1)
            acf = [1.0] + [float(np.corrcoef(s[:-k], s[k:])[0, 1]) for k in range(1, max_lag + 1)]
            ax.bar(lags, acf, color=chain_colors[c], alpha=0.7, width=0.8)
            ax.axhline(0, color="black", lw=0.8)
            ax.axhline(1.96 / np.sqrt(len(s)), color="grey", lw=0.8, ls="--")
            ax.axhline(-1.96 / np.sqrt(len(s)), color="grey", lw=0.8, ls="--")
            ax.set_ylim(-0.3, 1.05)
            ax.spines[["top","right"]].set_visible(False)
            if c == 0: ax.set_title(labels[var], fontsize=9)
            if j == 0: ax.set_ylabel(f"chain {c}", fontsize=8)
            if c == n_chains - 1: ax.set_xlabel("lag")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/autocorr.png", dpi=130, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved {out_dir}/autocorr.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading data…")
    df = load_data("top100.txt")
    print(df[["rank_biased", "song_title", "artist", "alpha_pos", "votes"]].head(10).to_string())

    model, idata, pos_mean, pos_std = build_and_sample(df)

    records, summary = build_results(df, idata, pos_mean, pos_std)
    save_results(records, summary)
    print("\nGenerating arviz diagnostic plots…")
    save_plots(idata)

    # Print top-10 unbiased
    print("\n=== Unbiased Top 10 ===")
    for r in records[:10]:
        delta_str = f"+{r['rank_delta']}" if r["rank_delta"] > 0 else str(r["rank_delta"])
        print(f"  {r['rank_unbiased']:>3}. {r['song_title']:<40} {r['artist']:<30} (was #{r['rank_biased']}, {delta_str})")

    # ── Counterfactual: what if Alberte were named Ålberte? ──────────────
    print("\n=== Counterfactual: Alberte → Ålberte ===")
    beta_mean = summary["beta_mean"]
    pos_mean_val = summary["pos_mean"]
    pos_std_val = summary["pos_std"]

    pos_A  = LETTER_RANK["A"]   # = 1
    pos_AA = LETTER_RANK["Å"]   # = 29

    # Standardised positions
    pos_A_std  = (pos_A  - pos_mean_val) / pos_std_val
    pos_AA_std = (pos_AA - pos_mean_val) / pos_std_val

    # Shift in log-votes when moving from A to Å
    delta_log_votes = beta_mean * (pos_AA_std - pos_A_std)

    # Find all Alberte songs in the results
    alberte_songs = [r for r in records if "Alberte" in r["artist"] and "Ålberte" not in r["artist"]]

    # Build a counterfactual ranking: adjust Alberte's corrected_log_votes
    alberte_keys = {(r["song_title"], r["artist"]) for r in alberte_songs}
    cf_records = []
    for r in records:
        if (r["song_title"], r["artist"]) in alberte_keys:
            cf_records.append({**r, "cf_quality": r["posterior_quality"] + delta_log_votes})
        else:
            cf_records.append({**r, "cf_quality": r["posterior_quality"]})

    cf_sorted = sorted(cf_records, key=lambda x: x["cf_quality"], reverse=True)
    for new_rank, r in enumerate(cf_sorted, start=1):
        if (r["song_title"], r["artist"]) in alberte_keys:
            print(f"  '{r['song_title']}' by Alberte:")
            print(f"    Original rank (biased):   #{r['rank_biased']}")
            print(f"    Unbiased rank (as-is):    #{r['rank_unbiased']}")
            print(f"    Counterfactual (Ålberte): #{new_rank}  (Δ log-votes = {delta_log_votes:+.4f})")
