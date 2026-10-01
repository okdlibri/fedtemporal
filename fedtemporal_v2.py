"""
FedTemporal v2: IEEE Journal Full Experimental Suite
=====================================================

Claim: "FedTemporal detects adaptive Byzantine attacks that exploit temporal
blind spots in per-round defenses."

Key upgrades over v1
--------------------
  Dataset  : CIFAR-10 (primary)  vs  MNIST (sanity check)
  Rounds   : 50                  vs  15
  Runs     : 5                   vs  3
  Ratios   : 0.2 / 0.3 / 0.4 / 0.5  vs  0.2 / 0.3 / 0.4

New attack suite (replace simple sleeper + oscillating)
  1. sleeper_mimicry      — honest warmup (60 %), then black-box gradient
                            mimicry aligned with server direction
  2. oscillating_strategic— period-4 oscillation (2 attack + 2 honest rounds);
                            longer honest window lets VFL-TS reputation recover
  3. gradient_drift       — gradual directional shift each round; below single-
                            round detection threshold until round ~15

New analysis
  • attack-phase accuracy  (accuracy only during active attack rounds)
  • convergence speed      (first round reaching target accuracy)
  • detection / false-positive rates  (FedTemporal-specific)
  • ablation study         (6 signal configurations)
  • hyperparameter sweep   (alpha, window_size)
  • Wilcoxon signed-rank tests  +  Cohen's d effect sizes
  • 8 publication-ready PDF plots

Usage
  python fedtemporal_v2.py               # full CIFAR-10 suite
  python fedtemporal_v2.py --sanity      # quick MNIST sanity check
  python fedtemporal_v2.py --ablation    # ablation study only
  python fedtemporal_v2.py --hyperparam  # hyperparameter sensitivity
"""

import os
import sys
import time
import json
import pickle
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from collections import deque, defaultdict
from scipy.stats import wilcoxon as scipy_wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# LOAD BASE IMPLEMENTATION  (temp-main-comp-V2.py)
# ============================================================================

_BASE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'temp-main-comp-V2.py'
)
print(f"Loading base implementation: {_BASE_FILE}")

_saved_name = __name__
__name__ = '_vflts_base_'
with open(_BASE_FILE, encoding='utf-8') as _fh:
    exec(compile(_fh.read(), _BASE_FILE, 'exec'))
__name__ = _saved_name
print("Base implementation loaded.\n")


# ============================================================================
# CONFIGURATION
# ============================================================================

config = CIFAR10Config()
config.NUM_RUNS        = 5
config.NUM_ROUNDS      = 50
config.DEBUG_MODE      = False
config.VERBOSE         = True
config.BASE_OUTPUT_DIR = './fedtemporal_results_v2'
config.OUTPUT_DIR      = './fedtemporal_results_v2'

# --- Algorithms --------------------------------------------------------------
V2_ALGORITHMS     = ['FedAvg', 'TrimmedMean', 'Krum', 'FLTrust', 'VFL_TS', 'FedTemporal']
NEW_BASELINES     = ['TrimmedMean', 'Krum', 'CoordinateMedian', 'FLAME', 'FLDetector']

# --- Attack types ------------------------------------------------------------
V2_ATTACK_TYPES = [
    'byzantine_signflip',    # reference: existing static attack
    'sleeper_mimicry',       # NEW: honest warmup + black-box mimicry
    'oscillating_strategic', # NEW: period-4 oscillation
    'gradient_drift',        # NEW: gradual directional rotation
    'lie',                   # INDEPENDENT: Little Is Enough (Baruch et al., NeurIPS 2019)
]

# LIE hyperparameters
LIE_AMPLIFICATION = 1.0     # multiplier on z*sigma perturbation (1.0 = paper default)

# --- Adversarial ratios ------------------------------------------------------
V2_ATTACK_RATIOS = [0.2, 0.3, 0.4, 0.5]

# --- Attack hyperparameters --------------------------------------------------
SLEEPER_WARMUP_FRACTION  = 0.60   # 30 of 50 rounds are honest warmup
OSCILLATING_PERIOD_V2    = 4      # 2 attack + 2 honest rounds per cycle
MIMICRY_POISON_STRENGTH  = 0.70   # 70 % poison energy, 30 % server-aligned
GRADIENT_DRIFT_RATE      = 1.0 / 30.0  # drift_factor → 0 after 30 rounds

_SLEEPER_WARMUP_V2 = max(1, int(config.NUM_ROUNDS * SLEEPER_WARMUP_FRACTION))

# --- FedTemporal hyperparameters ---------------------------------------------
FT_WINDOW_SIZE = 5
FT_ALPHA       = 0.5

# --- Ablation ----------------------------------------------------------------
ABLATION_VARIANTS = ['full', 'no_variance', 'no_trend', 'no_zscore',
                     'no_inertia', 'window1']
ABLATION_ATTACKS  = ['sleeper_mimicry', 'oscillating_strategic']
ABLATION_RATIO    = 0.4

# --- Hyperparameter sensitivity -----------------------------------------------
ALPHA_VALUES   = [0.3, 0.5, 0.7]
WINDOW_SIZES   = [3,   5,   10]
HYPERPARAM_ATTACK = 'sleeper_mimicry'
HYPERPARAM_RATIO  = 0.3
HYPERPARAM_RUNS   = 3

# --- Output ------------------------------------------------------------------
os.makedirs(config.BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(config.BASE_OUTPUT_DIR, 'plots'), exist_ok=True)
os.makedirs(os.path.join(config.BASE_OUTPUT_DIR, 'data'),  exist_ok=True)


# ============================================================================
# NEW ATTACKS
# ============================================================================

def apply_advanced_attack(weights, global_weights, prev_global_weights,
                          attack_type, round_num,
                          k_warmup=None, period=4,
                          poison_strength=0.70, drift_rate=None,
                          base_attack='signflip', amplification=3.0):
    """
    Advanced adaptive Byzantine attacks targeting per-round defense blind spots.

    sleeper_mimicry (black-box)
    ---------------------------
    Warmup  [0, k_warmup):  honest behaviour — builds reputation / trust history.
    Attack  [k_warmup, end): black-box mimicry.
      The attacker observes consecutive global model snapshots to estimate the
      aggregate server update direction d_srv = θ(t) − θ(t−1).  It then
      crafts a poisoned update that has a *positive* cosine similarity with
      d_srv (passing FLTrust's threshold of 0) while embedding (poison_strength)
      fraction of adversarial energy.

      d_attack = (1 − ps) * ||d_hon|| * d̂_srv  +  ps * (−d_honest)
               (server-aligned mimicry)            (sign-flip poison)

      • FLTrust:  cos(d_attack, d_srv) = (1−ps)*1 + ps*(−cos(d_hon,d_srv))
                  Still > 0 for ps < 0.5, accepted every round.
      • FedTemporal: Z-score anomaly fires — client suddenly behaves less like
                  its warmup history, triggering the temporal penalty.

    oscillating_strategic (period-4)
    ---------------------------------
    Period 0–1 (rounds where (round // period) % 2 == 0):  sign-flip attack.
    Period 2–3:  honest behaviour.
    The 4-round period (vs v1's 2-round) gives VFL-TS enough honest rounds to
    partially restore reputation, so the client never gets filtered for good.
    FedTemporal's variance anomaly catches the persistent flip-flop pattern.

    gradient_drift
    --------------
    No warmup.  Each round, the attacker smoothly interpolates from its honest
    update toward a sign-flip attack:
      drift_factor = max(0,  1 − round_num * drift_rate)   ∈ [1 → 0]
      d_attack = drift_factor * d_honest + (1 − drift_factor) * d_signflip
    cosine sim with server ≈ linearly declining over 30 rounds.
    FLTrust only filters when cos_sim < 0 (round ≥ ~15).
    FedTemporal's trend anomaly fires earlier (rounds 5–10).

    Args
    ----
    weights              : local model weights after honest training
    global_weights       : global model at start of current round θ(t)
    prev_global_weights  : global model from previous round θ(t−1), or None
    attack_type          : 'sleeper_mimicry' | 'oscillating_strategic' |
                           'gradient_drift' | 'byzantine_signflip'
    round_num            : current training round (0-indexed)
    k_warmup             : sleeper warmup length in rounds
    period               : oscillating cycle length in rounds
    poison_strength      : mimicry poison fraction (0 = fully mimicry, 1 = fully poison)
    drift_rate           : gradient_drift speed (1/round to full poison)
    base_attack          : underlying attack type for static component
    amplification        : Byzantine amplification factor

    Returns
    -------
    list of weight arrays (same structure as `weights`)
    """
    if k_warmup is None:
        k_warmup = max(1, int(config.NUM_ROUNDS * SLEEPER_WARMUP_FRACTION))
    if drift_rate is None:
        drift_rate = GRADIENT_DRIFT_RATE

    # ── Helpers ─────────────────────────────────────────────────────────────
    def _flat(wlist, ref):
        return np.concatenate([(w - g).flatten()
                                for w, g in zip(wlist, ref)])

    def _unflatten(d_flat, ref):
        out, offset = [], 0
        for g in ref:
            n = g.size
            out.append(g + d_flat[offset:offset + n].reshape(g.shape))
            offset += n
        return out

    # ── sleeper_mimicry ──────────────────────────────────────────────────────
    if attack_type == 'sleeper_mimicry':
        if round_num < k_warmup:
            return weights                                    # warmup: honest

        # Fall back to honest if no server direction available yet
        if prev_global_weights is None:
            return weights

        # Black-box server direction estimate: δ = θ(t) − θ(t−1)
        d_srv = np.concatenate([
            (g - p).flatten()
            for g, p in zip(global_weights, prev_global_weights)
        ])
        srv_norm = np.linalg.norm(d_srv)
        if srv_norm < 1e-10:
            return weights                                    # no movement, skip

        d_srv_unit  = d_srv / srv_norm

        # Honest update
        d_honest    = _flat(weights, global_weights)
        honest_norm = np.linalg.norm(d_honest)
        if honest_norm < 1e-10:
            return weights

        # Poison component (sign-flip of honest update)
        d_poison = -d_honest

        # Mimicry attack:
        #   (1 - ps) * honest_norm * d̂_srv  +  ps * d_poison
        d_attack = ((1.0 - poison_strength) * honest_norm * d_srv_unit
                    + poison_strength * d_poison)

        # Normalise back to honest norm to avoid anomalous magnitude
        atk_norm = np.linalg.norm(d_attack)
        if atk_norm > 1e-10:
            d_attack = d_attack * (honest_norm / atk_norm)

        return _unflatten(d_attack, global_weights)

    # ── oscillating_strategic ────────────────────────────────────────────────
    elif attack_type == 'oscillating_strategic':
        period_idx = round_num // period
        if period_idx % 2 == 0:                              # attack period
            return apply_byzantine_attack(
                weights, global_weights, base_attack, amplification
            )
        else:                                                 # honest period
            return weights

    # ── gradient_drift ───────────────────────────────────────────────────────
    elif attack_type == 'gradient_drift':
        drift_factor = max(0.0, 1.0 - round_num * drift_rate)

        d_honest   = _flat(weights, global_weights)
        honest_norm = np.linalg.norm(d_honest)
        if honest_norm < 1e-10:
            return weights

        # Get sign-flip attack update as target direction
        w_poison   = apply_byzantine_attack(
            weights, global_weights, base_attack, amplification
        )
        d_poison   = _flat(w_poison, global_weights)

        # Smooth interpolation: honest → signflip
        d_attack   = drift_factor * d_honest + (1.0 - drift_factor) * d_poison

        # Maintain the magnitude of the honest update
        atk_norm = np.linalg.norm(d_attack)
        if atk_norm > 1e-10:
            d_attack = d_attack * (honest_norm / atk_norm)

        return _unflatten(d_attack, global_weights)

    # ── static Byzantine (fallback) ──────────────────────────────────────────
    elif attack_type == 'byzantine_signflip':
        return apply_byzantine_attack(
            weights, global_weights, 'signflip', amplification
        )

    return weights   # safety fallback


# ============================================================================
# LIE ATTACK — Little Is Enough (Baruch et al., NeurIPS 2019)
# ============================================================================

def apply_lie_attack(weights, global_weights, n_clients, n_byzantine,
                     amplification=1.0):
    """
    Little Is Enough (LIE) — Baruch et al., NeurIPS 2019.
    Black-box adaptation: Byzantine clients shift their honest update by
    z * sigma in the anti-convergence direction, where z is calibrated to
    stay within the statistical 'safety zone' of defenses such as Trimmed
    Mean and Krum (which filter based on coordinate-wise deviation bounds).

    z = Phi^{-1}((n - f - 1) / (n - f))   (Gaussian inverse CDF)

    For n=20, f=6:   z ≈ 1.47
    For n=20, f=4:   z ≈ 1.96
    For n=20, f=8:   z ≈ 0.84

    The perturbation is in the direction that most opposes convergence
    (anti-honest-gradient direction), scaled to sigma * z where sigma is
    estimated from the attacker's own gradient norm.

    Key property: the attack is *temporally consistent* — each Byzantine
    client applies the same magnitude shift every round, so per-round
    variance is low.  This makes it a challenging test for temporal
    profiling defenses: FedTemporal's variance and Z-score signals may
    not fire if the shift is stable, but the trend signal should detect
    the systematic accuracy degradation over time.

    Args
    ----
    weights         : local model weights after honest training
    global_weights  : global model at start of round θ(t)
    n_clients       : total number of clients N
    n_byzantine     : number of Byzantine clients f (= int(N * ratio))
    amplification   : multiplier on z (>1 = more aggressive; default 1.0)

    Returns
    -------
    list of weight arrays (same structure as `weights`)
    """
    from scipy.stats import norm as sp_norm

    # Flatten to update-space vectors
    flat_g = np.concatenate([np.array(g).flatten() for g in global_weights])
    flat_w = np.concatenate([np.array(w).flatten() for w in weights])
    d_honest = flat_w - flat_g          # attacker's honest update vector

    d_norm = np.linalg.norm(d_honest)
    if d_norm < 1e-10:
        return weights                  # degenerate: nothing to corrupt

    # --- LIE z-parameter ---------------------------------------------------
    f = max(1, n_byzantine)
    n = max(f + 2, n_clients)           # ensure p is a valid probability
    p = (n - f - 1) / (n - f)          # Baruch et al. Eq. (3)
    p = float(np.clip(p, 0.501, 0.9999))
    z = float(sp_norm.ppf(p)) * amplification

    # --- Perturbation scale ------------------------------------------------
    # Coordinate-wise std proxy: ||d_honest|| / sqrt(d) (isotropic estimate)
    sigma_scalar = d_norm / (np.sqrt(len(d_honest)) + 1e-12)

    # --- Attack direction: anti-convergence (oppose honest update) ----------
    # LIE sets: d_attack = -mu + z*sigma  (anti-mean, within statistical bounds)
    # In black-box single-client form: -d_honest + z*sigma_scalar*d_unit
    # For large d, z*sigma_scalar/d_norm = z/sqrt(d) << 1, so this is
    # approximately sign-flip with a small correction that keeps it within
    # the safety zone of Trimmed Mean and Krum.
    d_unit   = d_honest / d_norm
    d_attack = -d_honest + z * sigma_scalar * d_unit

    # --- Reconstruct weight tensors ----------------------------------------
    attacked, idx = [], 0
    for g in global_weights:
        g_arr = np.array(g)
        size  = g_arr.size
        attacked.append(
            (flat_g[idx:idx + size] + d_attack[idx:idx + size]).reshape(g_arr.shape)
        )
        idx += size
    return attacked


# ============================================================================
# FEDTEMPORAL ALGORITHM  (v2 — with _compute_anomaly_score hook)
# ============================================================================

class FedTemporalAlgorithm(FederatedAlgorithm):
    """
    FedTemporal: Temporal Behavioral Profiling for Byzantine-Robust FL.

    Three key mechanisms over FLTrust:
      1. Sliding-window behavioral profiles per client
         Each round, records (cosine_similarity, norm_ratio) for the window.

      2. Temporal anomaly scoring
         Three complementary signals extract adaptive-attacker signatures:
           Signal A — Variance:  high cos-sim variance = unstable client
           Signal B — Trend:     declining slope = escalating attack
           Signal C — Z-score:   sudden departure from personal mean = sleeper trigger
         Combined into temporal_score ∈ [0, 1] via `_compute_anomaly_score()`.

      3. Trust inertia (exponential smoothing)
         combined_trust(t) = α * instant(t) * temporal(t) + (1−α) * combined(t−1)
         Prevents a recently honest client from instantly evading penalty.

    FLTrust is a special case: FedTemporalAblation(ablation='window1') ≡ FLTrust.
    """

    def __init__(self, input_shape, num_classes, root_dataset,
                 window_size=5, alpha=0.5):
        super().__init__('FedTemporal', input_shape, num_classes)
        self.root_dataset    = root_dataset
        self.window_size     = window_size
        self.alpha           = alpha

        self.profiles        = defaultdict(lambda: deque(maxlen=window_size))
        self.trust_scores    = {}
        self.current_round   = 0
        self.trust_log       = defaultdict(list)
        self.prev_global_weights = None   # set externally before each round

    # ── Client selection (random, same as FLTrust) ──────────────────────────

    def select_clients(self, round_num, client_data, k):
        available = [i for i in range(len(client_data))
                     if len(client_data[i][0]) > 0]
        return np.random.choice(
            available, size=min(k, len(available)), replace=False
        ).tolist()

    # ── Local training (attack injection handled externally) ─────────────────

    def train_client(self, client_idx, client_data, global_weights,
                     attack_type=None, is_malicious=False):
        X, y = client_data
        if len(X) == 0:
            return None, 0.0

        local_model = create_model(self.input_shape, self.num_classes)
        local_model.set_weights(global_weights)
        t0 = time.time()
        local_model.fit(X, y, epochs=config.EPOCHS_LOCAL,
                        batch_size=config.BATCH_SIZE, verbose=0)
        elapsed = time.time() - t0
        return local_model.get_weights(), elapsed

    # ── Anomaly scoring (overridable by ablation subclass) ───────────────────

    def _compute_anomaly_score(self, var_a, trend_a, zscore_a):
        """Combine three anomaly signals into a single score ∈ [0, 1].
        Subclasses override this to ablate individual signals."""
        return 0.40 * var_a + 0.30 * trend_a + 0.30 * zscore_a

    # ── Aggregation: the core contribution ──────────────────────────────────

    def aggregate(self, local_weights, client_indices, client_data):
        """
        FedTemporal trust-weighted aggregation (7-step pipeline).

        Step 1  Server reference update (FLTrust-style, root_dataset)
        Step 2  Instant trust  =  max(0, cosine(server_update, client_update))
        Step 3  Temporal anomaly score from sliding-window history
        Step 4  Update profiles (after anomaly scoring — non-self-referential)
        Step 5  Combined trust with inertia
        Step 6  Normalise + magnitude clip (FLTrust-style)
        Step 7  Weighted aggregation
        """
        if not local_weights:
            return None

        n  = len(local_weights)
        gw = self.global_model.get_weights()

        # ── Step 1: server reference update ──────────────────────────────────
        X_r, y_r = self.root_dataset
        srv = create_model(self.input_shape, self.num_classes)
        srv.set_weights(gw)
        srv.fit(X_r, y_r, epochs=config.EPOCHS_LOCAL,
                batch_size=config.BATCH_SIZE, verbose=0)
        srv_update = self._flat([sw - g for sw, g in zip(srv.get_weights(), gw)])
        srv_norm   = float(np.linalg.norm(srv_update))

        if srv_norm < 1e-10:
            return self._weighted_avg(local_weights, client_indices, client_data)

        # ── Step 2: instant trust (per-round cosine similarity) ───────────────
        instant   = []
        cli_norms = []

        for cw in local_weights:
            upd = self._flat([c - g for c, g in zip(cw, gw)])
            cn  = float(np.linalg.norm(upd))
            cos = (float(np.dot(srv_update, upd) / (srv_norm * cn))
                   if cn > 1e-10 else 0.0)
            instant.append(max(0.0, cos))
            cli_norms.append(cn)

        # ── Step 3: temporal anomaly scores from behavioral profiles ──────────
        temporal = []

        for i, cid in enumerate(client_indices):
            hist = self.profiles[cid]

            if len(hist) < 2:
                t_score = 1.0   # cold-start: no anomaly
            else:
                past_cos = [f['cos_sim'] for f in hist]

                # Signal A — Variance: high variation across window
                var_a = min(1.0, float(np.var(past_cos)) * 4.0)

                # Signal B — Trend: negative slope = declining alignment
                if len(past_cos) >= 3:
                    x     = np.arange(len(past_cos), dtype=float)
                    slope = float(np.polyfit(x, past_cos, 1)[0])
                else:
                    slope = 0.0
                trend_a = max(0.0, min(1.0, -slope * 10.0))

                # Signal C — Z-score: sudden departure from personal mean
                h_mean   = float(np.mean(past_cos))
                h_std    = max(float(np.std(past_cos)), 0.01)
                z        = abs(instant[i] - h_mean) / h_std
                zscore_a = min(1.0, z / 3.0)

                anomaly  = self._compute_anomaly_score(var_a, trend_a, zscore_a)
                t_score  = max(0.0, 1.0 - anomaly)

            temporal.append(t_score)

        # ── Step 4: update profiles (after scoring — non-self-referential) ────
        for i, cid in enumerate(client_indices):
            self.profiles[cid].append({
                'cos_sim'   : instant[i],
                'norm_ratio': cli_norms[i] / (srv_norm + 1e-10),
                'round'     : self.current_round,
            })

        # ── Step 5: combined trust with inertia ───────────────────────────────
        combined = []

        for i, cid in enumerate(client_indices):
            instant_comp = instant[i] * temporal[i]
            prev         = self.trust_scores.get(cid, instant_comp)
            c            = max(0.0, self.alpha * instant_comp
                               + (1.0 - self.alpha) * prev)
            combined.append(c)
            self.trust_scores[cid] = c

            self.trust_log[cid].append({
                'round'   : self.current_round,
                'instant' : instant[i],
                'temporal': temporal[i],
                'combined': c,
            })

        combined = np.array(combined)

        # ── Step 6: normalise + magnitude clip ───────────────────────────────
        tot = combined.sum()
        if tot < 1e-10:
            return self._weighted_avg(local_weights, client_indices, client_data)
        norm_w = combined / tot

        clipped = []
        for cw in local_weights:
            upd = [c - g for c, g in zip(cw, gw)]
            cn  = np.linalg.norm(self._flat(upd))
            if cn > srv_norm:
                upd = [u * (srv_norm / cn) for u in upd]
            clipped.append(upd)

        # ── Step 7: aggregate ─────────────────────────────────────────────────
        agg = []
        for li in range(len(gw)):
            layer = np.zeros_like(gw[li])
            for i in range(n):
                layer += norm_w[i] * clipped[i][li]
            agg.append(gw[li] + layer)

        self.current_round += 1
        return agg

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _flat(self, weights):
        return np.concatenate([w.flatten() for w in weights])

    def _weighted_avg(self, local_weights, client_indices, client_data):
        total = sum(len(client_data[i][1]) for i in client_indices)
        if total == 0:
            return None
        agg = []
        for li in range(len(local_weights[0])):
            layer = np.zeros_like(local_weights[0][li])
            for i, ci in enumerate(client_indices):
                layer += (len(client_data[ci][1]) / total) * local_weights[i][li]
            agg.append(layer)
        return agg


# ============================================================================
# FEDTEMPORAL ABLATION VARIANTS
# ============================================================================

class FedTemporalAblation(FedTemporalAlgorithm):
    """
    FedTemporal with configurable signal ablation.

    ablation options
    ----------------
    'full'        : all three signals active  (same as FedTemporalAlgorithm)
    'no_variance' : Signal A removed
    'no_trend'    : Signal B removed
    'no_zscore'   : Signal C removed  (disables sleeper-specific detection)
    'no_inertia'  : alpha = 1.0  (no trust memory, per-round evaluation)
    'window1'     : window_size = 1, alpha = 1.0  ≡ FLTrust exactly
    """

    _NAME_MAP = {
        'full'       : 'FedTemporal',
        'no_variance': 'FedTemporal-NoVar',
        'no_trend'   : 'FedTemporal-NoTrend',
        'no_zscore'  : 'FedTemporal-NoZ',
        'no_inertia' : 'FedTemporal-NoInertia',
        'window1'    : 'FedTemporal-W1',
    }

    def __init__(self, input_shape, num_classes, root_dataset,
                 window_size=5, alpha=0.5, ablation='full'):
        if ablation == 'window1':
            window_size, alpha = 1, 1.0
        elif ablation == 'no_inertia':
            alpha = 1.0
        super().__init__(input_shape, num_classes, root_dataset,
                         window_size, alpha)
        self.ablation = ablation
        self.name = self._NAME_MAP.get(ablation, 'FedTemporal')

    def _compute_anomaly_score(self, var_a, trend_a, zscore_a):
        abl = self.ablation
        if abl == 'no_variance': return 0.50 * trend_a + 0.50 * zscore_a
        if abl == 'no_trend':    return 0.57 * var_a   + 0.43 * zscore_a
        if abl == 'no_zscore':   return 0.57 * var_a   + 0.43 * trend_a
        # 'full', 'no_inertia', 'window1' all use the standard weights
        return 0.40 * var_a + 0.30 * trend_a + 0.30 * zscore_a


# ============================================================================
# EXTRA BASELINE ALGORITHMS  (not in temp-main-comp-V2.py)
# ============================================================================

class CoordinateMedianAlgorithm(FederatedAlgorithm):
    """Coordinate-wise Median: Byzantine-robust aggregation (Yin et al., ICML 2018)."""

    def __init__(self, input_shape, num_classes):
        super().__init__('CoordinateMedian', input_shape, num_classes)

    def select_clients(self, round_num, client_data, k):
        available = [i for i in range(len(client_data)) if len(client_data[i][0]) > 0]
        return np.random.choice(available, size=min(k, len(available)), replace=False).tolist()

    def train_client(self, client_idx, client_data, global_weights,
                     attack_type=None, is_malicious=False):
        X, y = client_data
        if len(X) == 0:
            return None, 0.0
        local_model = create_model(self.input_shape, self.num_classes)
        local_model.set_weights(global_weights)
        import time as _time
        t0 = _time.time()
        local_model.fit(X, y, epochs=config.EPOCHS_LOCAL,
                        batch_size=config.BATCH_SIZE, verbose=0)
        return local_model.get_weights(), _time.time() - t0

    def aggregate(self, local_weights, client_indices, client_data):
        if not local_weights:
            return None
        aggregated = []
        for layer_idx in range(len(local_weights[0])):
            layer_stack = np.array([w[layer_idx] for w in local_weights])
            aggregated.append(np.median(layer_stack, axis=0))
        return aggregated


class FLAMEAlgorithm(FederatedAlgorithm):
    """
    FLAME: Taming Backdoors in Federated Learning (Nguyen et al., USENIX 2022).
    Clusters updates with DBSCAN, clips to median norm, adds calibrated Gaussian noise.
    """

    def __init__(self, input_shape, num_classes, noise_multiplier=0.001, eps=0.5):
        super().__init__('FLAME', input_shape, num_classes)
        self.noise_multiplier = noise_multiplier
        self.eps = eps          # DBSCAN cosine-distance threshold

    def select_clients(self, round_num, client_data, k):
        available = [i for i in range(len(client_data)) if len(client_data[i][0]) > 0]
        return np.random.choice(available, size=min(k, len(available)), replace=False).tolist()

    def train_client(self, client_idx, client_data, global_weights,
                     attack_type=None, is_malicious=False):
        X, y = client_data
        if len(X) == 0:
            return None, 0.0
        local_model = create_model(self.input_shape, self.num_classes)
        local_model.set_weights(global_weights)
        import time as _time
        t0 = _time.time()
        local_model.fit(X, y, epochs=config.EPOCHS_LOCAL,
                        batch_size=config.BATCH_SIZE, verbose=0)
        return local_model.get_weights(), _time.time() - t0

    def aggregate(self, local_weights, client_indices, client_data):
        if not local_weights:
            return None

        from sklearn.cluster import DBSCAN
        from collections import Counter

        # Flatten each client's weights into a single vector
        flat = np.array([np.concatenate([w.flatten() for w in ws])
                         for ws in local_weights])

        # Pairwise cosine distance matrix
        norms = np.linalg.norm(flat, axis=1, keepdims=True)
        normed = flat / (norms + 1e-10)
        cos_dist = np.clip(1.0 - normed @ normed.T, 0.0, 2.0)

        # DBSCAN clustering — remove outlier cluster (label == -1)
        try:
            labels = DBSCAN(eps=self.eps, min_samples=2,
                            metric='precomputed').fit(cos_dist).labels_
            majority = Counter(l for l in labels if l != -1)
            if majority:
                best = majority.most_common(1)[0][0]
                kept = [i for i, l in enumerate(labels) if l == best]
            else:
                kept = list(range(len(local_weights)))
        except Exception:
            kept = list(range(len(local_weights)))

        kept_weights = [local_weights[i] for i in kept]
        kept_flat    = flat[kept]

        # Clip updates to median L2 norm
        client_norms  = np.linalg.norm(kept_flat, axis=1)
        median_norm   = float(np.median(client_norms))
        clipped = []
        for ws, n in zip(kept_weights, client_norms):
            scale = min(1.0, median_norm / (n + 1e-10))
            clipped.append([w * scale for w in ws])

        # Average + Gaussian noise calibrated to median_norm
        aggregated = []
        n_kept = len(clipped)
        for layer_idx in range(len(clipped[0])):
            avg = np.mean([w[layer_idx] for w in clipped], axis=0)
            sigma = self.noise_multiplier * median_norm / max(1.0, np.sqrt(n_kept))
            avg   = avg + np.random.normal(0.0, sigma, avg.shape)
            aggregated.append(avg)

        return aggregated


class FLDetectorAlgorithm(FederatedAlgorithm):
    """
    FLDetector: Defending FL Against Model Poisoning via Detecting Malicious Clients
    (Zhang et al., KDD 2022).  Flags clients whose gradient deviates significantly
    from a smooth-gradient prediction (consecutive-round inconsistency score).
    """

    def __init__(self, input_shape, num_classes, threshold=2.0):
        super().__init__('FLDetector', input_shape, num_classes)
        self.threshold = threshold          # MAD-based multiplier for adaptive cutoff
        self.prev_grads = {}                # client_id -> flat gradient from last round

    def select_clients(self, round_num, client_data, k):
        available = [i for i in range(len(client_data)) if len(client_data[i][0]) > 0]
        return np.random.choice(available, size=min(k, len(available)), replace=False).tolist()

    def train_client(self, client_idx, client_data, global_weights,
                     attack_type=None, is_malicious=False):
        X, y = client_data
        if len(X) == 0:
            return None, 0.0
        local_model = create_model(self.input_shape, self.num_classes)
        local_model.set_weights(global_weights)
        import time as _time
        t0 = _time.time()
        local_model.fit(X, y, epochs=config.EPOCHS_LOCAL,
                        batch_size=config.BATCH_SIZE, verbose=0)
        return local_model.get_weights(), _time.time() - t0

    def aggregate(self, local_weights, client_indices, client_data):
        if not local_weights:
            return None

        global_flat = np.concatenate([w.flatten()
                                      for w in self.global_model.get_weights()])

        scores = []
        for cid, ws in zip(client_indices, local_weights):
            grad = np.concatenate([w.flatten() for w in ws]) - global_flat
            prev = self.prev_grads.get(cid)
            if prev is not None:
                prev_norm = np.linalg.norm(prev)
                score = (np.linalg.norm(grad - prev) / (prev_norm + 1e-10))
            else:
                score = 0.0
            scores.append(score)
            self.prev_grads[cid] = grad.copy()

        scores = np.array(scores)
        median_s = np.median(scores)
        mad_s    = np.median(np.abs(scores - median_s))
        cutoff   = median_s + self.threshold * (mad_s + 1e-10)

        kept = [i for i, s in enumerate(scores) if s <= cutoff] or list(range(len(local_weights)))
        kept_weights = [local_weights[i] for i in kept]

        aggregated = []
        for layer_idx in range(len(kept_weights[0])):
            aggregated.append(np.mean([w[layer_idx] for w in kept_weights], axis=0))
        return aggregated


# ============================================================================
# ALGORITHM FACTORY
# ============================================================================

def make_algorithm(algo_name, shape, n_cls, root_dataset, shared_val,
                   ablation=None):
    """Instantiate the requested algorithm."""
    if algo_name == 'FedAvg':
        return FedAvgAlgorithm(shape, n_cls)
    elif algo_name == 'TrimmedMean':
        return TrimmedMeanAlgorithm(shape, n_cls)
    elif algo_name == 'Krum':
        return KrumAlgorithm(shape, n_cls)
    elif algo_name == 'CoordinateMedian':
        return CoordinateMedianAlgorithm(shape, n_cls)
    elif algo_name == 'FLAME':
        return FLAMEAlgorithm(shape, n_cls)
    elif algo_name == 'FLDetector':
        return FLDetectorAlgorithm(shape, n_cls)
    elif algo_name == 'FLTrust':
        return FLTrustAlgorithm(shape, n_cls, root_dataset)
    elif algo_name == 'VFL_TS':
        return VFLTSAlgorithm(shape, n_cls, shared_val)
    elif algo_name == 'FedTemporal' or algo_name.startswith('FedTemporal-'):
        if ablation and ablation != 'full':
            return FedTemporalAblation(
                shape, n_cls, root_dataset,
                window_size=FT_WINDOW_SIZE, alpha=FT_ALPHA,
                ablation=ablation
            )
        return FedTemporalAlgorithm(
            shape, n_cls, root_dataset,
            window_size=FT_WINDOW_SIZE, alpha=FT_ALPHA
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


# ============================================================================
# TRAINING LOOP  v3  (correct prev_global tracking + advanced attacks)
# ============================================================================

def run_federated_training_v3(algorithm, client_data, X_test, y_test,
                              attack_type=None, attack_ratio=0.0, run_id=0,
                              adaptive_params=None):
    """
    Extended training loop with:
      • Correct θ(t-1) tracking for black-box mimicry
      • All four attack types (static + three adaptive)
      • Extended history dict  (attack_active, trust_scores)

    Key invariant
    -------------
    At the START of round t:
      current_global      = θ(t)   (global model before this round's aggregation)
      prev_global_weights = θ(t-1) (global model from previous round)
    This gives the server direction estimate d_srv = θ(t) − θ(t-1).
    prev_global_weights is updated AFTER aggregation so the next round sees it.

    Returns
    -------
    history dict with keys:
      accuracy, loss, training_time, selected_clients, malicious_selected,
      kept_after_filtering, reputations, malicious_clients,
      trust_scores (FedTemporal only), attack_active
    """
    if adaptive_params is None:
        adaptive_params = {
            'k_warmup'       : max(1, int(config.NUM_ROUNDS * SLEEPER_WARMUP_FRACTION)),
            'period'         : OSCILLATING_PERIOD_V2,
            'poison_strength': MIMICRY_POISON_STRENGTH,
            'drift_rate'     : GRADIENT_DRIFT_RATE,
            'base_attack'    : 'signflip',
            'amplification'  : 3.0,
            'lie_amplification': LIE_AMPLIFICATION,
        }

    IS_ADVANCED = attack_type in (
        'sleeper_mimicry', 'oscillating_strategic', 'gradient_drift'
    )
    IS_LIE      = attack_type == 'lie'
    IS_STATIC   = attack_type == 'byzantine_signflip'

    print(f"\n{'='*68}")
    print(f"  Algorithm : {algorithm.name}")
    print(f"  Attack    : {attack_type or 'None'}   "
          f"Ratio: {attack_ratio}   Run: {run_id + 1}")
    if attack_type == 'sleeper_mimicry':
        kw = adaptive_params['k_warmup']
        print(f"  Warmup    : {kw} rounds  (attack starts round {kw + 1})")
    elif attack_type == 'oscillating_strategic':
        print(f"  Period    : {adaptive_params['period']} rounds")
    elif attack_type == 'gradient_drift':
        print(f"  Drift rate: {adaptive_params['drift_rate']:.4f} per round")
    elif attack_type == 'lie':
        print(f"  LIE amp   : {adaptive_params.get('lie_amplification', 1.0):.2f}  "
              f"(Baruch et al. NeurIPS 2019, black-box)")
    print(f"{'='*68}\n")

    # ── Initialise ────────────────────────────────────────────────────────────
    if hasattr(algorithm, 'initialize'):
        algorithm.initialize(len(client_data))
    else:
        algorithm.initialize_model()

    # ── Assign malicious clients ──────────────────────────────────────────────
    n_mal    = int(len(client_data) * attack_ratio)
    malicious = (set(np.random.choice(len(client_data), n_mal, replace=False))
                 if n_mal > 0 else set())

    if isinstance(algorithm, VFLTSAlgorithm):
        algorithm.malicious_clients = malicious

    if malicious:
        print(f"  Malicious clients: {sorted(malicious)}")

    # ── History ───────────────────────────────────────────────────────────────
    history = {
        'accuracy'            : [],
        'loss'                : [],
        'training_time'       : [],
        'selected_clients'    : [],
        'malicious_selected'  : [],
        'kept_after_filtering': [],
        'reputations'         : [],
        'malicious_clients'   : list(malicious),
        'trust_scores'        : [],   # FedTemporal only
        'attack_active'       : [],
    }

    # ── Training loop ─────────────────────────────────────────────────────────
    prev_global_weights = None   # θ(t-1)

    for round_num in range(config.NUM_ROUNDS):
        t0 = time.time()

        current_global = [w.copy() for w in algorithm.global_model.get_weights()]

        # ── Attack-active flag ────────────────────────────────────────────────
        if attack_type == 'sleeper_mimicry':
            attack_active = (round_num >= adaptive_params['k_warmup'])
        elif attack_type == 'oscillating_strategic':
            attack_active = ((round_num // adaptive_params['period']) % 2 == 0)
        elif attack_type == 'gradient_drift':
            attack_active = (attack_ratio > 0)   # always on, but mild early
        elif IS_STATIC or IS_LIE:
            attack_active = (attack_ratio > 0)
        else:
            attack_active = False

        # ── Client selection ──────────────────────────────────────────────────
        k        = max(1, int(len(client_data) * config.CLIENT_PARTICIPATION_RATE))
        selected = algorithm.select_clients(round_num, client_data, k)
        mal_this = [i for i in selected if i in malicious]

        if config.VERBOSE and (round_num + 1) % 10 == 0:
            _phase = ''
            if attack_type == 'sleeper_mimicry':
                _phase = (' [WARMUP]' if not attack_active
                          else f' [MIMICRY r{round_num - adaptive_params["k_warmup"] + 1}]')
            elif attack_type == 'oscillating_strategic':
                _phase = ' [ATTACKING]' if attack_active else ' [HONEST]'
            elif attack_type == 'gradient_drift':
                df = max(0.0, 1.0 - round_num * adaptive_params['drift_rate'])
                _phase = f' [DRIFT {df:.2f}]'
            elif attack_type == 'lie':
                _phase = f' [LIE z={adaptive_params.get("lie_amplification",1.0):.2f}]'
            print(f"  Round {round_num + 1:>3}/{config.NUM_ROUNDS}{_phase}  "
                  f"selected={len(selected)}  malicious={len(mal_this)}")

        # ── Local training + attack injection ─────────────────────────────────
        local_weights     = []
        validation_losses = []

        for cid in selected:
            is_mal = cid in malicious

            if isinstance(algorithm, VFLTSAlgorithm):
                w, _, vl = algorithm.train_client(
                    cid, client_data[cid], current_global, None, False
                )
                validation_losses.append(vl)
            else:
                w, _ = algorithm.train_client(
                    cid, client_data[cid], current_global, None, False
                )

            # Inject attack AFTER honest training
            if is_mal and w is not None:
                if IS_ADVANCED:
                    w = apply_advanced_attack(
                        w,
                        current_global,
                        prev_global_weights,
                        attack_type,
                        round_num,
                        k_warmup        = adaptive_params['k_warmup'],
                        period          = adaptive_params['period'],
                        poison_strength = adaptive_params['poison_strength'],
                        drift_rate      = adaptive_params['drift_rate'],
                        base_attack     = adaptive_params['base_attack'],
                        amplification   = adaptive_params['amplification'],
                    )
                elif IS_LIE:
                    w = apply_lie_attack(
                        w, current_global,
                        n_clients   = len(client_data),
                        n_byzantine = n_mal,
                        amplification = adaptive_params.get('lie_amplification', 1.0),
                    )
                elif IS_STATIC:
                    w = apply_byzantine_attack(
                        w, current_global, 'signflip',
                        adaptive_params['amplification']
                    )

            if w is not None:
                local_weights.append(w)

        # ── Aggregation ───────────────────────────────────────────────────────
        kept = list(selected)
        if local_weights:
            if isinstance(algorithm, VFLTSAlgorithm):
                agg = algorithm.aggregate(
                    local_weights, selected,
                    validation_losses, client_data, round_num
                )
            else:
                agg = algorithm.aggregate(local_weights, selected, client_data)

            if agg:
                algorithm.global_model.set_weights(agg)

        # VFL-TS periodic reputation boost
        if (round_num > 0 and round_num % 3 == 0
                and isinstance(algorithm, VFLTSAlgorithm)):
            algorithm.reputation_system.boost_consistent_performers(round_num)

        # CRITICAL: advance prev_global AFTER aggregation
        prev_global_weights = [w.copy() for w in current_global]

        # ── Evaluation ────────────────────────────────────────────────────────
        test_loss, test_acc = algorithm.global_model.evaluate(
            X_test, y_test, verbose=0
        )

        # ── Record ────────────────────────────────────────────────────────────
        history['accuracy'].append(float(test_acc))
        history['loss'].append(float(test_loss))
        history['training_time'].append(time.time() - t0)
        history['selected_clients'].append(list(selected))
        history['malicious_selected'].append(mal_this)
        history['kept_after_filtering'].append(kept)
        history['attack_active'].append(bool(attack_active))

        if isinstance(algorithm, VFLTSAlgorithm):
            history['reputations'].append(
                algorithm.reputation_system.get_all_reputations().copy()
            )
        else:
            history['reputations'].append(np.zeros(len(client_data)))

        if isinstance(algorithm, FedTemporalAlgorithm):
            history['trust_scores'].append(
                {cid: algorithm.trust_scores.get(cid, 0.0)
                 for cid in range(len(client_data))}
            )
        else:
            history['trust_scores'].append({})

    print(f"  Final accuracy: {history['accuracy'][-1]:.4f}")
    return history


# ============================================================================
# METRICS
# ============================================================================

def compute_attack_phase_accuracy(history):
    """Mean accuracy during attack-active rounds only."""
    active = [i for i, a in enumerate(history['attack_active']) if a]
    if not active:
        return float('nan')
    return float(np.mean([history['accuracy'][i] for i in active]))


def compute_convergence_round(history, target_accuracy=0.70):
    """First round where accuracy >= target_accuracy. Returns NUM_ROUNDS if never."""
    for i, acc in enumerate(history['accuracy']):
        if acc >= target_accuracy:
            return i + 1
    return config.NUM_ROUNDS


def compute_detection_metrics(history, malicious_clients, trust_threshold=0.3):
    """
    FedTemporal-specific detection metrics.

    detection_rate    : fraction of malicious clients with combined_trust < threshold,
                        averaged over attack-active rounds.
    false_positive_rate: fraction of honest clients with combined_trust < threshold,
                         averaged over ALL rounds.
    """
    if not history['trust_scores'] or not history['trust_scores'][0]:
        return {'detection_rate': float('nan'), 'false_positive_rate': float('nan')}

    n_clients = max(max(ts.keys()) for ts in history['trust_scores'] if ts) + 1
    honest_set    = set(range(n_clients)) - set(malicious_clients)
    malicious_set = set(malicious_clients)

    active_rounds = [i for i, a in enumerate(history['attack_active']) if a]
    all_rounds    = list(range(len(history['trust_scores'])))

    def _rate(client_set, rounds):
        if not client_set or not rounds:
            return float('nan')
        vals = []
        for r in rounds:
            ts = history['trust_scores'][r]
            detected = sum(1 for cid in client_set
                           if ts.get(cid, 1.0) < trust_threshold)
            vals.append(detected / len(client_set))
        return float(np.mean(vals))

    return {
        'detection_rate'    : _rate(malicious_set, active_rounds),
        'false_positive_rate': _rate(honest_set,    all_rounds),
    }


def aggregate_runs(runs, metric='accuracy'):
    """Return (mean_per_round, std_per_round) arrays over multiple runs."""
    arrays = np.array([r[metric] for r in runs])
    return arrays.mean(axis=0), arrays.std(axis=0)


def final_accuracy_stats(runs, last_k=5):
    """Mean and std of final accuracy, averaged over last k rounds per run."""
    finals = [float(np.mean(r['accuracy'][-last_k:])) for r in runs]
    return float(np.mean(finals)), float(np.std(finals))


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def cohen_d(a, b):
    """Cohen's d effect size between two paired samples."""
    diff  = np.array(a) - np.array(b)
    pooled_std = np.std(diff, ddof=1)
    return float(np.mean(diff) / pooled_std) if pooled_std > 1e-10 else 0.0


def run_wilcoxon_tests(results, target='FedTemporal', baselines=None,
                       last_k=5):
    """
    Paired Wilcoxon signed-rank tests: target vs each baseline.

    For every (attack, ratio) combination, extracts final accuracy
    (mean of last k rounds) across runs and tests target vs baseline.

    Returns
    -------
    dict  {(baseline, attack, ratio): {p_value, effect_size, n_runs, sig}}
    """
    if baselines is None:
        baselines = [a for a in V2_ALGORITHMS if a != target]

    out = {}
    if target not in results:
        return out

    for baseline in baselines:
        if baseline not in results:
            continue
        for atk in V2_ATTACK_TYPES:
            for ratio in V2_ATTACK_RATIOS:
                rkey = f'ratio_{ratio}'
                try:
                    tgt_runs = results[target][atk][rkey]
                    bas_runs = results[baseline][atk][rkey]
                except KeyError:
                    continue
                if len(tgt_runs) < 2 or len(bas_runs) < 2:
                    continue

                n = min(len(tgt_runs), len(bas_runs))
                t_vals = [float(np.mean(r['accuracy'][-last_k:]))
                          for r in tgt_runs[:n]]
                b_vals = [float(np.mean(r['accuracy'][-last_k:]))
                          for r in bas_runs[:n]]

                try:
                    stat, p = scipy_wilcoxon(t_vals, b_vals, zero_method='wilcox')
                except Exception:
                    p = float('nan')

                d = cohen_d(t_vals, b_vals)
                key = (baseline, atk, ratio)
                out[key] = {
                    'p_value'     : float(p),
                    'effect_size' : d,
                    'n_runs'      : n,
                    'significant' : (p < 0.05) if not np.isnan(p) else False,
                    'target_mean' : float(np.mean(t_vals)),
                    'baseline_mean': float(np.mean(b_vals)),
                }

    return out


def print_wilcoxon_table(wtest):
    """Print a readable summary of Wilcoxon test results."""
    print("\n" + "=" * 90)
    print("WILCOXON SIGNED-RANK TESTS  (FedTemporal vs baselines)")
    print(f"{'Baseline':<14} {'Attack':<24} {'Ratio':>6}  "
          f"{'FT Mean':>9}  {'Base Mean':>9}  {'p-value':>9}  {'d':>6}  {'Sig':>4}")
    print("-" * 90)
    for (bl, atk, ratio), v in sorted(wtest.items()):
        sig = '***' if v['p_value'] < 0.001 else ('**' if v['p_value'] < 0.01
              else ('*' if v['p_value'] < 0.05 else ''))
        print(f"{bl:<14} {atk:<24} {ratio:>6.1f}  "
              f"{v['target_mean']:>9.4f}  {v['baseline_mean']:>9.4f}  "
              f"{v['p_value']:>9.4f}  {v['effect_size']:>6.3f}  {sig:>4}")
    print("=" * 90)
    print("  * p<0.05   ** p<0.01   *** p<0.001\n")


# ============================================================================
# RESULTS I/O
# ============================================================================

def _to_serialisable(obj):
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, dict):        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_serialisable(x) for x in obj]
    if isinstance(obj, bool):        return obj
    return obj


def save_results(results, tag='main'):
    data_dir = os.path.join(config.BASE_OUTPUT_DIR, 'data')
    os.makedirs(data_dir, exist_ok=True)

    pkl_path  = os.path.join(data_dir, f'fedtemporal_v2_{tag}.pkl')
    json_path = os.path.join(data_dir, f'fedtemporal_v2_{tag}.json')

    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(_to_serialisable(results), f, indent=2)

    print(f"  [Saved → {data_dir}/fedtemporal_v2_{tag}.{{pkl,json}}]")


# ============================================================================
# EXPERIMENT RUNNERS
# ============================================================================

def _load_data_and_roots(cfg=None):
    """Load dataset, split for clients, build shared_val and root_dataset."""
    (X_train, y_train), (X_test, y_test), n_cls, shape = load_dataset(config.DATASET)
    client_data = create_non_iid_split(
        X_train, y_train, config.NUM_CLIENTS, config.NON_IID_ALPHA
    )
    shared_val  = construct_shared_validation_set(
        client_data,
        config.SHARED_VAL_SAMPLES_PER_CLIENT,
        config.SHARED_VAL_MAX_SIZE
    )
    Xsv, ysv    = shared_val
    root_n      = max(100, int(len(Xsv) * 0.3))
    root_dataset = (Xsv[:root_n], ysv[:root_n])
    return (X_test, y_test), n_cls, shape, client_data, shared_val, root_dataset


def run_experiment_grid(algorithms, attack_types, ratios, num_runs,
                        n_cls, shape, client_data, shared_val, root_dataset,
                        X_test, y_test, tag='main', ablation_map=None):
    """
    Generic experiment grid runner.

    ablation_map : {algo_name: ablation_variant}  for ablation study
    """
    if ablation_map is None:
        ablation_map = {}

    adaptive_params = {
        'k_warmup'       : max(1, int(config.NUM_ROUNDS * SLEEPER_WARMUP_FRACTION)),
        'period'         : OSCILLATING_PERIOD_V2,
        'poison_strength': MIMICRY_POISON_STRENGTH,
        'drift_rate'     : GRADIENT_DRIFT_RATE,
        'base_attack'    : 'signflip',
        'amplification'  : 3.0,
    }

    total  = len(algorithms) * len(attack_types) * len(ratios) * num_runs
    count  = 0
    results = {}

    for algo_name in algorithms:
        results[algo_name] = {}
        abl = ablation_map.get(algo_name)

        for atk in attack_types:
            results[algo_name][atk] = {}

            for ratio in ratios:
                rkey = f'ratio_{ratio}'
                results[algo_name][atk][rkey] = []

                for run_id in range(num_runs):
                    count += 1
                    print(f"\n[{count}/{total}] "
                          f"{algo_name} | {atk} | ratio={ratio} | run={run_id + 1}")

                    import tensorflow as tf
                    np.random.seed(config.RANDOM_SEED + run_id)
                    tf.random.set_seed(config.RANDOM_SEED + run_id)

                    algo = make_algorithm(
                        algo_name, shape, n_cls,
                        root_dataset, shared_val,
                        ablation=abl
                    )
                    hist = run_federated_training_v3(
                        algo, client_data, X_test, y_test,
                        attack_type    = atk,
                        attack_ratio   = ratio,
                        run_id         = run_id,
                        adaptive_params= adaptive_params,
                    )
                    results[algo_name][atk][rkey].append(hist)

        # Crash-safe save after each algorithm
        save_results(results, tag=tag)

    return results


def run_sanity_check():
    """MNIST, 15 rounds, 3 runs — validates v2 code against v1 results."""
    print("\n" + "=" * 68)
    print("  SANITY CHECK  (MNIST, 15 rounds, 3 runs)")
    print("=" * 68)

    orig_dataset = config.DATASET
    orig_rounds  = config.NUM_ROUNDS
    orig_runs    = config.NUM_RUNS

    config.DATASET    = 'MNIST'
    config.NUM_ROUNDS = 15
    config.NUM_RUNS   = 3

    (X_test, y_test), n_cls, shape, client_data, shared_val, root_dataset \
        = _load_data_and_roots()

    results = run_experiment_grid(
        algorithms   = V2_ALGORITHMS,
        attack_types = ['byzantine_signflip', 'sleeper_mimicry'],
        ratios       = [0.3],
        num_runs     = 3,
        n_cls=n_cls, shape=shape,
        client_data=client_data, shared_val=shared_val,
        root_dataset=root_dataset, X_test=X_test, y_test=y_test,
        tag='sanity',
    )

    config.DATASET    = orig_dataset
    config.NUM_ROUNDS = orig_rounds
    config.NUM_RUNS   = orig_runs

    print_results_table(results, ['byzantine_signflip', 'sleeper_mimicry'],
                        [0.3], V2_ALGORITHMS)
    return results


def run_lie_experiments():
    """
    Focused LIE attack experiment (Baruch et al., NeurIPS 2019).
    CIFAR-10, 50 rounds, 5 runs, all 9 algorithms, ratios 0.2–0.5.
    Results saved to fedtemporal_results_lie/ for inclusion in the paper
    as an independent-attack validation.
    """
    print("\n" + "=" * 68)
    print("  LIE ATTACK EXPERIMENT  (Baruch et al., NeurIPS 2019)")
    print("  CIFAR-10, 50 rounds, 5 runs — independent attack validation")
    print(f"  Algorithms : {V2_ALGORITHMS + NEW_BASELINES}")
    print(f"  Ratios     : {V2_ATTACK_RATIOS}")
    print("=" * 68)

    orig_outdir = config.BASE_OUTPUT_DIR
    config.BASE_OUTPUT_DIR = './fedtemporal_results_lie'
    os.makedirs(config.BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.BASE_OUTPUT_DIR, 'data'),  exist_ok=True)
    os.makedirs(os.path.join(config.BASE_OUTPUT_DIR, 'plots'), exist_ok=True)

    (X_test, y_test), n_cls, shape, client_data, shared_val, root_dataset \
        = _load_data_and_roots()

    all_algos = V2_ALGORITHMS + NEW_BASELINES

    results = run_experiment_grid(
        algorithms   = all_algos,
        attack_types = ['lie'],
        ratios       = V2_ATTACK_RATIOS,
        num_runs     = config.NUM_RUNS,
        n_cls=n_cls, shape=shape,
        client_data=client_data, shared_val=shared_val,
        root_dataset=root_dataset, X_test=X_test, y_test=y_test,
        tag='lie',
    )

    config.BASE_OUTPUT_DIR = orig_outdir
    print_results_table(results, ['lie'], V2_ATTACK_RATIOS, all_algos)
    return results


# ============================================================================
# NSL-KDD IIoT DATASET SUPPORT
# ============================================================================
# NSL-KDD is the standard benchmark for network intrusion detection in IIoT.
# It represents traffic logs from network sensors in industrial control systems.
# Reference: Tavallaee et al., "A detailed analysis of the KDD CUP 99 data
# set", IEEE Symposium on Computational Intelligence for Security and Defense
# Applications, 2009.
#
# Natural FL split: 20 network monitoring sensors, one per service category.
# Each sensor monitors a specific type of network service (HTTP, FTP, SMTP …),
# creating naturally heterogeneous traffic distributions across clients.
# ============================================================================

NSLKDD_TRAIN_URL  = ('https://raw.githubusercontent.com/defcom17/NSL_KDD'
                     '/master/KDDTrain%2B.txt')
NSLKDD_TEST_URL   = ('https://raw.githubusercontent.com/defcom17/NSL_KDD'
                     '/master/KDDTest%2B.txt')
NSLKDD_CACHE      = './data_cache/nslkdd'
NSLKDD_N_CLIENTS  = 20          # one FL sensor per top-20 service category
NSLKDD_ROUNDS     = 30          # 30 communication rounds
NSLKDD_N_CLASSES  = 4           # normal, DoS, Probe, R2L/U2R
NSLKDD_RATIOS     = [0.2, 0.3, 0.4]
NSLKDD_ATTACKS    = V2_ATTACK_TYPES   # all 5 attacks
NSLKDD_RUNS       = 5

# GPU split: divide V2_ALGORITHMS across two GPUs to halve total runtime
# Part A → GPU assigned via CUDA_VISIBLE_DEVICES=0
# Part B → GPU assigned via CUDA_VISIBLE_DEVICES=1
IIOT_ALGOS_A = ['FedAvg', 'TrimmedMean', 'Krum']
IIOT_ALGOS_B = ['FLTrust', 'VFL_TS', 'FedTemporal']

# Map 23 raw attack labels → 4 classes
# 0: normal   1: DoS   2: Probe   3: R2L / U2R
_NSLKDD_LABEL_MAP = {
    'normal': 0,
    # --- DoS ---
    'neptune': 1, 'smurf': 1, 'back': 1, 'teardrop': 1, 'pod': 1,
    'land': 1, 'mailbomb': 1, 'processtable': 1, 'udpstorm': 1, 'apache2': 1,
    # --- Probe ---
    'ipsweep': 2, 'nmap': 2, 'portsweep': 2, 'satan': 2,
    'mscan': 2, 'saint': 2,
    # --- R2L / U2R (combined due to small R2L/U2R sample counts) ---
    'ftp_write': 3, 'guess_passwd': 3, 'imap': 3, 'multihop': 3, 'phf': 3,
    'spy': 3, 'warezclient': 3, 'warezmaster': 3, 'snmpgetattack': 3,
    'named': 3, 'xlock': 3, 'xsnoop': 3, 'sendmail': 3, 'httptunnel': 3,
    'snmpguess': 3, 'worm': 3, 'buffer_overflow': 3, 'ps': 3,
    'rootkit': 3, 'loadmodule': 3, 'perl': 3, 'xterm': 3, 'sqlattack': 3,
}

# Categorical feature encoding (fit on train split)
_NSLKDD_PROTOCOL_MAP = {'tcp': 0, 'udp': 1, 'icmp': 2}
_NSLKDD_FLAG_MAP = {}    # filled by load_nslkdd_dataset()
_NSLKDD_SERVICE_MAP = {} # filled by load_nslkdd_dataset()


def _parse_nslkdd_lines(raw_text, fit=False):
    """
    Parse raw NSL-KDD text (comma-separated) into (X, y, service_col).

    Columns layout (1-indexed):
      1        : duration         (numeric)
      2        : protocol_type   (categorical: tcp/udp/icmp)
      3        : service          (categorical: http, ftp, smtp …)
      4        : flag             (categorical: SF, S0, REJ …)
      5–41     : 37 numeric features
      42       : class label      (normal, neptune, …)
      43       : difficulty level (dropped)

    Returns X (float32, N × 41), y (int32, N), service_ids (int32, N).
    """
    global _NSLKDD_FLAG_MAP, _NSLKDD_SERVICE_MAP

    rows = [line.split(',') for line in raw_text.strip().split('\n') if line.strip()]
    protocols = [r[1] for r in rows]
    services  = [r[2] for r in rows]
    flags     = [r[3] for r in rows]
    labels    = [r[41] for r in rows]

    if fit:
        _NSLKDD_SERVICE_MAP = {s: i for i, s in enumerate(sorted(set(services)))}
        _NSLKDD_FLAG_MAP    = {f: i for i, f in enumerate(sorted(set(flags)))}

    n = len(rows)
    X = np.zeros((n, 41), dtype=np.float32)
    # Feature 0: duration (col 0)
    # Feature 1: protocol_type encoded
    # Feature 2: service encoded
    # Feature 3: flag encoded
    # Features 4–41: numeric columns 4–41 from original
    for i, r in enumerate(rows):
        X[i, 0]  = float(r[0])                                          # duration
        X[i, 1]  = float(_NSLKDD_PROTOCOL_MAP.get(r[1], 0))
        X[i, 2]  = float(_NSLKDD_SERVICE_MAP.get(r[2], 0))
        X[i, 3]  = float(_NSLKDD_FLAG_MAP.get(r[3], 0))
        for j in range(4, 41):
            try:
                X[i, j] = float(r[j])
            except (ValueError, IndexError):
                X[i, j] = 0.0

    y         = np.array([_NSLKDD_LABEL_MAP.get(r[41], 3) for r in rows], dtype=np.int32)
    svc_ids   = np.array([_NSLKDD_SERVICE_MAP.get(r[2], 0) for r in rows], dtype=np.int32)
    return X, y, svc_ids


def load_nslkdd_dataset(cache_dir=NSLKDD_CACHE):
    """
    Download (once, ~22 MB from GitHub) and load the NSL-KDD dataset.

    Returns
    -------
    (X_train, y_train, svc_train), (X_test, y_test)
      X: float32 (N, 41)  — 41 engineered network-flow features
      y: int32   (N,)     — 4-class labels: 0=normal, 1=DoS, 2=Probe, 3=R2L/U2R
      svc_train: int32 service-category IDs (used for client split)
    """
    import urllib.request, pathlib

    cache = pathlib.Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    def _cached_fetch(url, fname):
        dst = cache / fname
        if not dst.exists():
            print(f'Downloading NSL-KDD {fname} from GitHub...')
            urllib.request.urlretrieve(url, str(dst))
            print(f'  → saved {dst.stat().st_size / 1024 / 1024:.1f} MB')
        return dst.read_text(encoding='utf-8')

    raw_train = _cached_fetch(NSLKDD_TRAIN_URL, 'KDDTrain+.txt')
    raw_test  = _cached_fetch(NSLKDD_TEST_URL,  'KDDTest+.txt')

    X_tr, y_tr, svc_tr = _parse_nslkdd_lines(raw_train, fit=True)
    X_te, y_te, _      = _parse_nslkdd_lines(raw_test,  fit=False)

    # Normalize using training statistics
    mu  = X_tr.mean(axis=0)
    sig = X_tr.std(axis=0) + 1e-8
    X_tr = ((X_tr - mu) / sig).astype(np.float32)
    X_te = ((X_te - mu) / sig).astype(np.float32)

    print(f'NSL-KDD loaded: train={len(X_tr)}, test={len(X_te)}, '
          f'services={len(_NSLKDD_SERVICE_MAP)}, classes={NSLKDD_N_CLASSES}')
    print(f'  Class dist (train): {np.bincount(y_tr)}')
    return (X_tr, y_tr, svc_tr), (X_te, y_te)


def create_nslkdd_client_split(X_train, y_train, svc_ids, n_clients=NSLKDD_N_CLIENTS):
    """
    Split training data by service category — one FL client per service type.

    Each client = one network monitoring sensor observing a specific service
    (HTTP monitor, FTP monitor, SMTP monitor, etc.).  This creates natural
    non-IID distributions because different services carry very different
    attack profiles (e.g., HTTP sees mostly normal + Probe; FTP sees more R2L).
    """
    # Pick top-n_clients services by sample count
    unique_svcs, counts = np.unique(svc_ids, return_counts=True)
    top_svcs = unique_svcs[np.argsort(-counts)[:n_clients]]

    client_data = []
    svc_inv = {v: k for k, v in _NSLKDD_SERVICE_MAP.items()}
    for sid in top_svcs:
        mask  = (svc_ids == sid)
        X_c, y_c = X_train[mask], y_train[mask]
        client_data.append((X_c, y_c))
        svc_name = svc_inv.get(sid, str(sid))
        print(f'  Service {svc_name:12s}: {len(X_c):6d} samples, '
              f'class dist={np.bincount(y_c, minlength=NSLKDD_N_CLASSES)}')

    n_samples = [len(c[0]) for c in client_data]
    print(f'NSL-KDD split: {len(client_data)} clients, '
          f'{np.mean(n_samples):.0f} ± {np.std(n_samples):.0f} samples/client')
    return client_data


def _nslkdd_create_model(input_shape, num_classes, model_type='AUTO'):
    """
    3-layer MLP for 41-dimensional NSL-KDD network-flow features.
    Temporarily replaces the base create_model() during IIoT experiments.
    """
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def run_iiot_experiments(part=None):
    """
    NSL-KDD IIoT network intrusion detection experiment.

    Setup
    -----
    • 20 FL clients, one per top-20 network service category
    • Each client = a network monitoring sensor on one service subnet
    • 4 traffic classes: normal, DoS, Probe, R2L/U2R
    • 41 engineered network-flow features per sample
    • 30 communication rounds, 5 independent runs
    • All 5 attacks, Byzantine ratios [0.2, 0.3, 0.4]

    part : None | 'A' | 'B'
        None — run all 6 algorithms sequentially on one GPU  (~450 runs)
        'A'  — run IIOT_ALGOS_A = ['FedAvg', 'TrimmedMean', 'Krum']
               output → fedtemporal_results_v2_iiot_A/
        'B'  — run IIOT_ALGOS_B = ['FLTrust', 'VFL_TS', 'FedTemporal']
               output → fedtemporal_results_v2_iiot_B/
        A and B run in parallel on different GPUs, then merged with:
               python fedtemporal_v2.py --merge-iiot

    IIoT Relevance
    --------------
    NSL-KDD represents network intrusion detection for industrial IoT:
    compromised IIoT sensors can be modeled as Byzantine FL clients that
    report falsified traffic summaries to evade centralized IDS.
    FedTemporal's temporal behavioral profiling detects such clients by
    tracking how their reported traffic statistics diverge over time from
    consistent honest sensors.
    """
    # Determine which algorithms to run and output directory for this part
    if part == 'A':
        algos   = IIOT_ALGOS_A
        out_dir = './fedtemporal_results_v2_iiot_A'
        tag     = 'iiot_A'
    elif part == 'B':
        algos   = IIOT_ALGOS_B
        out_dir = './fedtemporal_results_v2_iiot_B'
        tag     = 'iiot_B'
    else:
        algos   = V2_ALGORITHMS
        out_dir = './fedtemporal_results_v2_iiot'
        tag     = 'iiot'

    part_label = f' — Part {part}  ({algos})' if part else ''
    print('\n' + '=' * 68)
    print(f'  IIoT EXPERIMENT  (NSL-KDD — 20 network sensors, 30 rounds){part_label}')
    print('  Dataset  : NSL-KDD Network Intrusion Detection (41 features, 4 classes)')
    print('  Clients  : 20 (one per service category — natural non-IID by service)')
    print(f'  Algorithms: {algos}')
    print(f'  Attacks  : {NSLKDD_ATTACKS}')
    print(f'  Ratios   : {NSLKDD_RATIOS}')
    print('=' * 68)

    # ── Save original state (restored in finally) ────────────────────────────
    _orig_outdir   = config.BASE_OUTPUT_DIR
    _orig_clients  = config.NUM_CLIENTS
    _orig_rounds   = config.NUM_ROUNDS
    _orig_dataset  = config.DATASET
    _orig_create_m = globals()['create_model']
    _orig_drift    = globals()['GRADIENT_DRIFT_RATE']

    # ── Patch global state for NSL-KDD ───────────────────────────────────────
    config.BASE_OUTPUT_DIR        = out_dir
    config.NUM_CLIENTS            = NSLKDD_N_CLIENTS
    config.NUM_ROUNDS             = NSLKDD_ROUNDS
    config.DATASET                = 'NSL_KDD'
    # Use MLP suited for 41-d features; base classes look up 'create_model' in globals
    globals()['create_model']     = _nslkdd_create_model
    # Faster drift: full poison by round 18 (60% of 30), consistent with CIFAR-10 ratio
    globals()['GRADIENT_DRIFT_RATE'] = 1.0 / (NSLKDD_ROUNDS * 0.60)

    os.makedirs(config.BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.BASE_OUTPUT_DIR, 'data'),  exist_ok=True)
    os.makedirs(os.path.join(config.BASE_OUTPUT_DIR, 'plots'), exist_ok=True)

    try:
        # ── Load NSL-KDD ─────────────────────────────────────────────────────
        (X_tr, y_tr, svc_tr), (X_te, y_te) = load_nslkdd_dataset()

        print('\nCreating service-based FL client split (natural non-IID by service):')
        client_data = create_nslkdd_client_split(X_tr, y_tr, svc_tr)

        n_cls = NSLKDD_N_CLASSES
        shape = (41,)

        shared_val   = construct_shared_validation_set(
            client_data,
            config.SHARED_VAL_SAMPLES_PER_CLIENT,
            config.SHARED_VAL_MAX_SIZE,
        )
        Xsv, ysv     = shared_val
        root_n       = max(100, int(len(Xsv) * 0.3))
        root_dataset = (Xsv[:root_n], ysv[:root_n])

        # ── Run experiment grid for this part ────────────────────────────────
        results = run_experiment_grid(
            algorithms   = algos,
            attack_types = NSLKDD_ATTACKS,
            ratios       = NSLKDD_RATIOS,
            num_runs     = NSLKDD_RUNS,
            n_cls        = n_cls,
            shape        = shape,
            client_data  = client_data,
            shared_val   = shared_val,
            root_dataset = root_dataset,
            X_test       = X_te,
            y_test       = y_te,
            tag          = tag,
        )

    finally:
        # ── Always restore original state ────────────────────────────────────
        globals()['create_model']        = _orig_create_m
        globals()['GRADIENT_DRIFT_RATE'] = _orig_drift
        config.BASE_OUTPUT_DIR = _orig_outdir
        config.NUM_CLIENTS     = _orig_clients
        config.NUM_ROUNDS      = _orig_rounds
        config.DATASET         = _orig_dataset

    print_results_table(results, NSLKDD_ATTACKS, NSLKDD_RATIOS, algos)
    return results


def merge_iiot_parts():
    """
    Merge Part A and Part B IIoT results into a single combined file.
    Run this after both --iiot-A and --iiot-B have completed.
    """
    path_A = './fedtemporal_results_v2_iiot_A/data/fedtemporal_v2_iiot_A.json'
    path_B = './fedtemporal_results_v2_iiot_B/data/fedtemporal_v2_iiot_B.json'

    # Temporarily point output to the merged folder
    orig_outdir = config.BASE_OUTPUT_DIR
    config.BASE_OUTPUT_DIR = './fedtemporal_results_v2_iiot'
    os.makedirs('./fedtemporal_results_v2_iiot/data', exist_ok=True)

    merged = merge_results(path_A, path_B, out_tag='iiot')
    config.BASE_OUTPUT_DIR = orig_outdir

    print_results_table(merged, NSLKDD_ATTACKS, NSLKDD_RATIOS, V2_ALGORITHMS)
    print(f'\n  Merged results saved to ./fedtemporal_results_v2_iiot/data/fedtemporal_v2_iiot.json')
    return merged


def run_main_experiments():
    """CIFAR-10, 50 rounds, 5 runs — full journal results."""
    print("\n" + "=" * 68)
    print("  MAIN EXPERIMENTS  (CIFAR-10, 50 rounds, 5 runs)")
    print(f"  Algorithms : {V2_ALGORITHMS}")
    print(f"  Attacks    : {V2_ATTACK_TYPES}")
    print(f"  Ratios     : {V2_ATTACK_RATIOS}")
    print(f"  Sleeper warmup : {_SLEEPER_WARMUP_V2} rounds "
          f"({int(SLEEPER_WARMUP_FRACTION * 100)}% of {config.NUM_ROUNDS})")
    print("=" * 68)

    (X_test, y_test), n_cls, shape, client_data, shared_val, root_dataset \
        = _load_data_and_roots()

    results = run_experiment_grid(
        algorithms   = V2_ALGORITHMS,
        attack_types = V2_ATTACK_TYPES,
        ratios       = V2_ATTACK_RATIOS,
        num_runs     = config.NUM_RUNS,
        n_cls=n_cls, shape=shape,
        client_data=client_data, shared_val=shared_val,
        root_dataset=root_dataset, X_test=X_test, y_test=y_test,
        tag='main',
    )
    return results


def run_ablation_experiments():
    """Ablation study: 6 FedTemporal variants on sleeper_mimicry + oscillating_strategic, ratio=0.3."""
    print("\n" + "=" * 68)
    print("  ABLATION STUDY")
    print(f"  Variants : {ABLATION_VARIANTS}")
    print(f"  Attacks  : {ABLATION_ATTACKS}   Ratio: {ABLATION_RATIO}")
    print("=" * 68)

    (X_test, y_test), n_cls, shape, client_data, shared_val, root_dataset \
        = _load_data_and_roots()

    # Map variant names → (algo_name, ablation)
    algo_names  = [f'FedTemporal-{v}' if v != 'full' else 'FedTemporal'
                   for v in ABLATION_VARIANTS]
    ablation_map = {f'FedTemporal-{v}': v for v in ABLATION_VARIANTS if v != 'full'}
    ablation_map['FedTemporal'] = 'full'

    results = run_experiment_grid(
        algorithms   = algo_names,
        attack_types = ABLATION_ATTACKS,
        ratios       = [ABLATION_RATIO],
        num_runs     = config.NUM_RUNS,
        n_cls=n_cls, shape=shape,
        client_data=client_data, shared_val=shared_val,
        root_dataset=root_dataset, X_test=X_test, y_test=y_test,
        tag='ablation',
        ablation_map = ablation_map,
    )
    return results


def run_hyperparam_sensitivity():
    """Sweep alpha and window_size on sleeper_mimicry, ratio=0.3."""
    print("\n" + "=" * 68)
    print("  HYPERPARAMETER SENSITIVITY")
    print(f"  Alpha values   : {ALPHA_VALUES}")
    print(f"  Window sizes   : {WINDOW_SIZES}")
    print(f"  Attack         : {HYPERPARAM_ATTACK}   Ratio: {HYPERPARAM_RATIO}")
    print("=" * 68)

    import tensorflow as tf
    (X_test, y_test), n_cls, shape, client_data, shared_val, root_dataset \
        = _load_data_and_roots()

    adaptive_params = {
        'k_warmup': max(1, int(config.NUM_ROUNDS * SLEEPER_WARMUP_FRACTION)), 'period': OSCILLATING_PERIOD_V2,
        'poison_strength': MIMICRY_POISON_STRENGTH, 'drift_rate': GRADIENT_DRIFT_RATE,
        'base_attack': 'signflip', 'amplification': 3.0,
    }

    results = {'alpha': {}, 'window': {}}

    for alpha in ALPHA_VALUES:
        key = f'alpha_{alpha}'
        results['alpha'][key] = []
        for run_id in range(HYPERPARAM_RUNS):
            np.random.seed(config.RANDOM_SEED + run_id)
            tf.random.set_seed(config.RANDOM_SEED + run_id)
            algo = FedTemporalAlgorithm(shape, n_cls, root_dataset,
                                        window_size=FT_WINDOW_SIZE, alpha=alpha)
            hist = run_federated_training_v3(
                algo, client_data, X_test, y_test,
                attack_type=HYPERPARAM_ATTACK, attack_ratio=HYPERPARAM_RATIO,
                run_id=run_id, adaptive_params=adaptive_params,
            )
            results['alpha'][key].append(hist)

    for ws in WINDOW_SIZES:
        key = f'window_{ws}'
        results['window'][key] = []
        for run_id in range(HYPERPARAM_RUNS):
            np.random.seed(config.RANDOM_SEED + run_id)
            tf.random.set_seed(config.RANDOM_SEED + run_id)
            algo = FedTemporalAlgorithm(shape, n_cls, root_dataset,
                                        window_size=ws, alpha=FT_ALPHA)
            hist = run_federated_training_v3(
                algo, client_data, X_test, y_test,
                attack_type=HYPERPARAM_ATTACK, attack_ratio=HYPERPARAM_RATIO,
                run_id=run_id, adaptive_params=adaptive_params,
            )
            results['window'][key].append(hist)

    save_results(results, tag='hyperparam')
    return results


# ============================================================================
# CONSOLE RESULTS TABLE
# ============================================================================

def run_baseline():
    """CIFAR-10, no attack — establishes the accuracy ceiling for all 4 algorithms."""
    print("\n" + "=" * 68)
    print("  BASELINE  (CIFAR-10, no attack, 5 runs)")
    print(f"  Algorithms : {V2_ALGORITHMS}")
    print("=" * 68)

    import tensorflow as tf
    (X_test, y_test), n_cls, shape, client_data, shared_val, root_dataset \
        = _load_data_and_roots()

    results = {}
    total = len(V2_ALGORITHMS) * config.NUM_RUNS
    count = 0

    for algo_name in V2_ALGORITHMS:
        results[algo_name] = []
        for run_id in range(config.NUM_RUNS):
            count += 1
            print(f"\n[{count}/{total}] {algo_name} | no attack | run={run_id + 1}")
            np.random.seed(config.RANDOM_SEED + run_id)
            tf.random.set_seed(config.RANDOM_SEED + run_id)
            algo = make_algorithm(algo_name, shape, n_cls, root_dataset, shared_val)
            hist = run_federated_training_v3(
                algo, client_data, X_test, y_test,
                attack_type=None, attack_ratio=0.0, run_id=run_id,
            )
            results[algo_name].append(hist)

        # Crash-safe save after each algorithm
        save_results(results, tag='baseline')

    print("\n" + "=" * 68)
    print("BASELINE RESULTS  (mean ± std of final accuracy, last 5 rounds)")
    print(f"{'Algorithm':<20}  {'Mean Acc':>10}  {'Std':>8}  {'Conv. Rd':>9}")
    print("-" * 52)
    for algo_name, runs in results.items():
        if not runs:
            continue
        mu, std = final_accuracy_stats(runs, last_k=5)
        conv = float(np.mean([compute_convergence_round(r) for r in runs]))
        print(f"{algo_name:<20}  {mu:>10.4f}  {std:>8.4f}  {conv:>9.1f}")
    print("=" * 68 + "\n")

    return results


def run_fmnist_experiments():
    """
    Fashion-MNIST validation suite — provides second-dataset evidence for the paper.

    Scope (lighter than CIFAR-10 main suite to save compute):
      - 4 algorithms, 2 key attacks (sleeper_mimicry + gradient_drift), ratios [0.2, 0.3, 0.4]
      - Plus a clean baseline (no attack, ratio=0.0)
      - 5 runs, 30 rounds  (uses FashionMNISTConfig defaults)
    """
    print("\n" + "=" * 68)
    print("  FASHION-MNIST VALIDATION")
    print(f"  Algorithms : {V2_ALGORITHMS}")
    print(f"  Attacks    : sleeper_mimicry, gradient_drift  (+baseline)")
    print(f"  Ratios     : [0.2, 0.3, 0.4]")
    print("=" * 68)

    # --- swap config ---
    fmnist_cfg = FashionMNISTConfig()
    fmnist_cfg.NUM_RUNS        = 5
    fmnist_cfg.DEBUG_MODE      = False
    fmnist_cfg.VERBOSE         = True
    fmnist_cfg.BASE_OUTPUT_DIR = './fedtemporal_results_v2'

    # Temporarily replace global config fields
    orig_dataset   = config.DATASET
    orig_rounds    = config.NUM_ROUNDS
    orig_runs      = config.NUM_RUNS
    orig_clients   = config.NUM_CLIENTS
    orig_output    = config.BASE_OUTPUT_DIR
    orig_epochs    = config.EPOCHS_LOCAL
    orig_batch     = config.BATCH_SIZE
    orig_part_rate = config.CLIENT_PARTICIPATION_RATE
    orig_noniid    = config.NON_IID_ALPHA
    orig_shared_per = config.SHARED_VAL_SAMPLES_PER_CLIENT
    orig_shared_max = config.SHARED_VAL_MAX_SIZE

    config.DATASET                       = fmnist_cfg.DATASET
    config.NUM_ROUNDS                    = fmnist_cfg.NUM_ROUNDS
    config.NUM_RUNS                      = fmnist_cfg.NUM_RUNS
    config.NUM_CLIENTS                   = fmnist_cfg.NUM_CLIENTS
    config.EPOCHS_LOCAL                  = fmnist_cfg.EPOCHS_LOCAL
    config.BATCH_SIZE                    = fmnist_cfg.BATCH_SIZE
    config.CLIENT_PARTICIPATION_RATE     = fmnist_cfg.CLIENT_PARTICIPATION_RATE
    config.NON_IID_ALPHA                 = fmnist_cfg.NON_IID_ALPHA
    config.SHARED_VAL_SAMPLES_PER_CLIENT = fmnist_cfg.SHARED_VAL_SAMPLES_PER_CLIENT
    config.SHARED_VAL_MAX_SIZE           = fmnist_cfg.SHARED_VAL_MAX_SIZE

    print(f"  Dataset  : {config.DATASET}   Rounds: {config.NUM_ROUNDS}   "
          f"Clients: {config.NUM_CLIENTS}   Runs: {config.NUM_RUNS}\n")

    (X_test, y_test), n_cls, shape, client_data, shared_val, root_dataset \
        = _load_data_and_roots()

    # Attack grid
    results = run_experiment_grid(
        algorithms   = V2_ALGORITHMS,
        attack_types = ['sleeper_mimicry', 'gradient_drift'],
        ratios       = [0.2, 0.3, 0.4],
        num_runs     = config.NUM_RUNS,
        n_cls=n_cls, shape=shape,
        client_data=client_data, shared_val=shared_val,
        root_dataset=root_dataset, X_test=X_test, y_test=y_test,
        tag='fmnist',
    )

    # Baseline (no attack)
    import tensorflow as tf
    total = len(V2_ALGORITHMS) * config.NUM_RUNS
    count = 0
    baseline_results = {}
    for algo_name in V2_ALGORITHMS:
        baseline_results[algo_name] = []
        for run_id in range(config.NUM_RUNS):
            count += 1
            print(f"\n[{count}/{total}] {algo_name} | no attack | run={run_id + 1}")
            np.random.seed(config.RANDOM_SEED + run_id)
            tf.random.set_seed(config.RANDOM_SEED + run_id)
            algo = make_algorithm(algo_name, shape, n_cls, root_dataset, shared_val)
            hist = run_federated_training_v3(
                algo, client_data, X_test, y_test,
                attack_type=None, attack_ratio=0.0, run_id=run_id,
            )
            baseline_results[algo_name].append(hist)
        save_results(baseline_results, tag='fmnist_baseline')

    # --- restore config ---
    config.DATASET                       = orig_dataset
    config.NUM_ROUNDS                    = orig_rounds
    config.NUM_RUNS                      = orig_runs
    config.NUM_CLIENTS                   = orig_clients
    config.BASE_OUTPUT_DIR               = orig_output
    config.EPOCHS_LOCAL                  = orig_epochs
    config.BATCH_SIZE                    = orig_batch
    config.CLIENT_PARTICIPATION_RATE     = orig_part_rate
    config.NON_IID_ALPHA                 = orig_noniid
    config.SHARED_VAL_SAMPLES_PER_CLIENT = orig_shared_per
    config.SHARED_VAL_MAX_SIZE           = orig_shared_max

    print_results_table(results, ['sleeper_mimicry', 'gradient_drift'],
                        [0.2, 0.3, 0.4], V2_ALGORITHMS)
    return results, baseline_results


def run_new_baselines():
    """
    Run ONLY the 5 new baseline algorithms on the full CIFAR-10 attack grid.
    Results are saved separately then merged with the existing main results.
    Avoids re-running the 4 already-completed algorithms.
    """
    print("\n" + "=" * 68)
    print("  NEW BASELINES  (CIFAR-10, 50 rounds, 5 runs)")
    print(f"  Algorithms : {NEW_BASELINES}")
    print(f"  Attacks    : {V2_ATTACK_TYPES}")
    print(f"  Ratios     : {V2_ATTACK_RATIOS}")
    print("=" * 68)

    (X_test, y_test), n_cls, shape, client_data, shared_val, root_dataset \
        = _load_data_and_roots()

    results = run_experiment_grid(
        algorithms   = NEW_BASELINES,
        attack_types = V2_ATTACK_TYPES,
        ratios       = V2_ATTACK_RATIOS,
        num_runs     = config.NUM_RUNS,
        n_cls=n_cls, shape=shape,
        client_data=client_data, shared_val=shared_val,
        root_dataset=root_dataset, X_test=X_test, y_test=y_test,
        tag='new_baselines',
    )
    return results


def merge_results(primary_path, secondary_path, out_tag='merged'):
    """
    Merge two result JSONs (keyed by algorithm name) into one combined dict.
    Saves to BASE_OUTPUT_DIR/data/fedtemporal_v2_{out_tag}.json.
    Returns the merged dict.
    """
    with open(primary_path) as f:
        primary = json.load(f)
    with open(secondary_path) as f:
        secondary = json.load(f)

    merged = dict(primary)
    for algo, data in secondary.items():
        if algo in merged:
            print(f"  [merge] WARNING: '{algo}' exists in both files — keeping primary.")
        else:
            merged[algo] = data
            print(f"  [merge] Added '{algo}' from secondary.")

    save_results(merged, tag=out_tag)
    print(f"  [merge] Saved merged results as tag='{out_tag}'")
    return merged


def print_results_table(results, attack_types, ratios, algorithms, last_k=5):
    print("\n" + "=" * 85)
    print("RESULTS TABLE  (mean ± std of final accuracy, last 5 rounds)")
    print(f"{'Algorithm':<20} {'Attack':<24} {'Ratio':>6}  "
          f"{'Mean Acc':>10}  {'Std':>8}  {'Atk-Phase':>10}  {'Conv. Rd':>9}")
    print("-" * 85)

    for algo in algorithms:
        if algo not in results:
            continue
        for atk in attack_types:
            if atk not in results.get(algo, {}):
                continue
            for ratio in ratios:
                rkey = f'ratio_{ratio}'
                runs = results[algo].get(atk, {}).get(rkey, [])
                if not runs:
                    continue
                mu, std = final_accuracy_stats(runs, last_k)
                atk_acc = float(np.mean([compute_attack_phase_accuracy(r) for r in runs]))
                conv    = float(np.mean([compute_convergence_round(r) for r in runs]))
                flag    = ' <<' if (algo == 'FedTemporal'
                                    and atk != 'byzantine_signflip') else ''
                print(f"{algo:<20} {atk:<24} {ratio:>6.1f}  "
                      f"{mu:>10.4f}  {std:>8.4f}  {atk_acc:>10.4f}  "
                      f"{conv:>9.1f}{flag}")
        print()

    print("=" * 85)
    print("(<< marks FedTemporal on adaptive attacks — key paper results)\n")


# ============================================================================
# PLOTS
# ============================================================================

_COLORS = {
    'FedAvg'     : '#e74c3c',
    'FLTrust'    : '#3498db',
    'VFL_TS'     : '#2ecc71',
    'FedTemporal': '#9b59b6',
}
_LINES = {
    'FedAvg'     : '--',
    'FLTrust'    : '-.',
    'VFL_TS'     : ':',
    'FedTemporal': '-',
}
_LW = {
    'FedAvg'     : 1.5,
    'FLTrust'    : 2.0,
    'VFL_TS'     : 2.0,
    'FedTemporal': 2.5,
}

_ATTACK_LABELS = {
    'byzantine_signflip'    : 'Byzantine\n(sign-flip)',
    'sleeper_mimicry'       : 'Sleeper\n(mimicry)',
    'oscillating_strategic' : 'Oscillating\n(strategic)',
    'gradient_drift'        : 'Gradient\nDrift',
}


def _save_fig(filename):
    path = os.path.join(config.BASE_OUTPUT_DIR, 'plots', filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {filename}")


def plot_learning_curves(results):
    """
    One row per attack type, one column per ratio.
    Mean accuracy ± 1 std shaded; attack timing annotated.
    """
    print("  Generating learning curve plots …")

    for atk in V2_ATTACK_TYPES:
        n_ratios = len(V2_ATTACK_RATIOS)
        fig, axes = plt.subplots(1, n_ratios, figsize=(4 * n_ratios, 4.5), sharey=True)
        if n_ratios == 1:
            axes = [axes]
        fig.suptitle(f'Attack: {atk.replace("_", " ").title()}  |  {config.DATASET}',
                     fontsize=13, fontweight='bold')

        for ax_i, ratio in enumerate(V2_ATTACK_RATIOS):
            ax   = axes[ax_i]
            rkey = f'ratio_{ratio}'

            for algo in V2_ALGORITHMS:
                runs = results.get(algo, {}).get(atk, {}).get(rkey, [])
                if not runs:
                    continue
                mu, sigma = aggregate_runs(runs)
                rds = np.arange(1, len(mu) + 1)
                ax.plot(rds, mu, label=algo,
                        color=_COLORS.get(algo, 'gray'),
                        linestyle=_LINES.get(algo, '-'),
                        linewidth=_LW.get(algo, 1.5))
                ax.fill_between(rds, mu - sigma, mu + sigma,
                                color=_COLORS.get(algo, 'gray'), alpha=0.10)

            # Attack timing annotations
            if atk == 'sleeper_mimicry':
                kw = max(1, int(config.NUM_ROUNDS * SLEEPER_WARMUP_FRACTION))
                ax.axvline(x=kw, color='firebrick', linestyle=':', linewidth=1.5,
                           label=f'Attack start (r={kw})', alpha=0.8)
                ax.axvspan(kw, config.NUM_ROUNDS + 1, alpha=0.04, color='red')

            elif atk == 'oscillating_strategic':
                first = True
                for r0 in range(0, config.NUM_ROUNDS, OSCILLATING_PERIOD_V2 * 2):
                    ax.axvspan(r0 + 1,
                               min(r0 + OSCILLATING_PERIOD_V2,
                                   config.NUM_ROUNDS) + 1,
                               alpha=0.07, color='firebrick',
                               label='Attack phase' if first else '')
                    first = False

            elif atk == 'gradient_drift':
                # Shade the drift-active region (rounds 1-30)
                ax.axvspan(1, min(31, config.NUM_ROUNDS + 1),
                           alpha=0.06, color='darkorange',
                           label='Drift active (rounds 1-30)')

            ax.set_title(f'Ratio: {int(ratio * 100)}%', fontsize=11)
            ax.set_xlabel('Round', fontsize=10)
            if ax_i == 0:
                ax.set_ylabel('Test Accuracy', fontsize=10)
            ax.set_ylim([0.0, 1.05])
            ax.legend(fontsize=7, loc='lower right')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        _save_fig(f'curves_{atk}.pdf')


def plot_summary_bar(results):
    """
    Two grouped bar charts side-by-side:
      Left  — final accuracy (last 5 rounds)
      Right — attack-phase accuracy
    One group per attack type, one bar per algorithm, at ratio=0.3.
    """
    print("  Generating summary bar chart …")

    ratio = 0.3
    rkey  = f'ratio_{ratio}'
    algos = V2_ALGORITHMS
    atks  = V2_ATTACK_TYPES
    x     = np.arange(len(atks))
    width = 0.7 / len(algos)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax_i, (metric_fn, ylabel, fname_suf) in enumerate([
        (lambda runs: final_accuracy_stats(runs)[0],       'Mean Final Accuracy', 'final'),
        (lambda runs: float(np.nanmean([compute_attack_phase_accuracy(r) for r in runs])),
         'Attack-Phase Accuracy', 'atk_phase'),
    ]):
        ax = axes[ax_i]
        for i, algo in enumerate(algos):
            means, stds = [], []
            for atk in atks:
                runs = results.get(algo, {}).get(atk, {}).get(rkey, [])
                if runs:
                    val = metric_fn(runs)
                    means.append(val)
                    stds.append(float(np.std([compute_attack_phase_accuracy(r)
                                              for r in runs]))
                                if ax_i == 1
                                else final_accuracy_stats(runs)[1])
                else:
                    means.append(0.0); stds.append(0.0)

            offset = (i - len(algos) / 2 + 0.5) * width
            ax.bar(x + offset, means, width * 0.92,
                   label=algo, color=_COLORS.get(algo, 'gray'),
                   yerr=stds, capsize=3, alpha=0.85)

        ax.set_xlabel('Attack Type', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(
            f'{ylabel}  (ratio={int(ratio * 100)}%,  {config.DATASET})',
            fontsize=11, fontweight='bold'
        )
        ax.set_xticks(x)
        ax.set_xticklabels([_ATTACK_LABELS.get(a, a) for a in atks], fontsize=9)
        ax.legend(fontsize=8)
        ax.set_ylim([0.0, 1.05])
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    _save_fig('summary_bar.pdf')


def plot_attack_phase_accuracy(results):
    """
    Heatmap: algorithm × attack type, colour = attack-phase accuracy at ratio=0.3.
    Highlights where each defense struggles most.
    """
    print("  Generating attack-phase accuracy heatmap …")

    ratio = 0.3
    rkey  = f'ratio_{ratio}'
    data  = np.zeros((len(V2_ALGORITHMS), len(V2_ATTACK_TYPES)))

    for i, algo in enumerate(V2_ALGORITHMS):
        for j, atk in enumerate(V2_ATTACK_TYPES):
            runs = results.get(algo, {}).get(atk, {}).get(rkey, [])
            if runs:
                data[i, j] = float(np.nanmean(
                    [compute_attack_phase_accuracy(r) for r in runs]
                ))

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=[_ATTACK_LABELS.get(a, a) for a in V2_ATTACK_TYPES],
                yticklabels=V2_ALGORITHMS,
                vmin=0.0, vmax=1.0, ax=ax, linewidths=0.5)
    ax.set_title(f'Attack-Phase Accuracy  (ratio={int(ratio*100)}%,  {config.DATASET})',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Attack Type', fontsize=11)
    ax.set_ylabel('Algorithm', fontsize=11)
    plt.tight_layout()
    _save_fig('heatmap_attack_phase.pdf')


def plot_trust_evolution(results):
    """
    FedTemporal trust trajectories: malicious vs honest per adaptive attack.
    Run 0, ratio=0.3 — single representative run for paper figure.
    """
    print("  Generating trust evolution plots …")

    ratio = 0.3
    rkey  = f'ratio_{ratio}'

    for atk in ['sleeper_mimicry', 'oscillating_strategic', 'gradient_drift']:
        runs = results.get('FedTemporal', {}).get(atk, {}).get(rkey, [])
        if not runs:
            continue

        run      = runs[0]
        ts_hist  = run.get('trust_scores', [])
        malicious= set(run.get('malicious_clients', []))
        n_clients= config.NUM_CLIENTS
        rounds   = list(range(1, len(ts_hist) + 1))

        if not ts_hist:
            continue

        fig, ax = plt.subplots(figsize=(10, 4.5))
        plotted_mal = plotted_hon = False

        for cid in range(n_clients):
            vals = [ts_hist[r].get(cid, 0.0) for r in range(len(rounds))]
            if cid in malicious:
                lbl = 'Malicious' if not plotted_mal else ''
                ax.plot(rounds, vals, color='crimson', alpha=0.75,
                        linewidth=1.8, label=lbl)
                plotted_mal = True
            else:
                lbl = 'Honest' if not plotted_hon else ''
                ax.plot(rounds, vals, color='steelblue', alpha=0.25,
                        linewidth=0.8, label=lbl)
                plotted_hon = True

        # Attack timing
        if atk == 'sleeper_mimicry':
            kw = max(1, int(config.NUM_ROUNDS * SLEEPER_WARMUP_FRACTION))
            ax.axvline(x=kw, color='darkred', linestyle='--', linewidth=2.0,
                       label=f'Mimicry starts (r={kw})')
            ax.axvspan(kw, config.NUM_ROUNDS + 1, alpha=0.04, color='red')

        elif atk == 'oscillating_strategic':
            first = True
            for r0 in range(0, config.NUM_ROUNDS, OSCILLATING_PERIOD_V2 * 2):
                ax.axvspan(r0 + 1,
                           min(r0 + OSCILLATING_PERIOD_V2, config.NUM_ROUNDS) + 1,
                           alpha=0.08, color='red',
                           label='Attack phase' if first else '')
                first = False

        elif atk == 'gradient_drift':
            ax.axvspan(1, min(31, config.NUM_ROUNDS + 1),
                       alpha=0.06, color='darkorange', label='Drift region')

        ax.set_xlabel('Round', fontsize=11)
        ax.set_ylabel('Combined Trust Score', fontsize=11)
        ax.set_title(
            f'FedTemporal Trust Scores — {atk.replace("_", " ").title()}  '
            f'(ratio={int(ratio * 100)}%,  {config.DATASET})',
            fontsize=12, fontweight='bold'
        )
        ax.set_ylim([0.0, 1.05])
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save_fig(f'trust_evolution_{atk}.pdf')


def plot_ablation(ablation_results):
    """Two-panel bar chart comparing 6 FedTemporal variants on both ablation attacks."""
    print("  Generating ablation plot …")

    rkey = f'ratio_{ABLATION_RATIO}'
    display_order = ['FedTemporal', 'FedTemporal-NoVar', 'FedTemporal-NoTrend',
                     'FedTemporal-NoZ', 'FedTemporal-NoInertia', 'FedTemporal-W1']

    fig, axes = plt.subplots(1, len(ABLATION_ATTACKS), figsize=(7 * len(ABLATION_ATTACKS), 4.5))
    if len(ABLATION_ATTACKS) == 1:
        axes = [axes]

    for ax, atk in zip(axes, ABLATION_ATTACKS):
        names, means, stds = [], [], []
        for algo in display_order:
            runs = ablation_results.get(algo, {}).get(atk, {}).get(rkey, [])
            if not runs:
                continue
            mu, std = final_accuracy_stats(runs)
            names.append(algo.replace('FedTemporal-', '-'))
            means.append(mu)
            stds.append(std)

        if not means:
            continue

        colors = ['#9b59b6'] + ['#bdc3c7'] * (len(names) - 1)
        bars = ax.bar(names, means, color=colors, yerr=stds, capsize=4, alpha=0.88,
                      edgecolor='black', linewidth=0.5)
        bars[0].set_edgecolor('#6c3483')
        bars[0].set_linewidth(2.0)

        ax.set_ylabel('Mean Final Accuracy', fontsize=11)
        ax.set_xlabel('FedTemporal Variant', fontsize=11)
        ax.set_title(
            f'Ablation  —  {atk.replace("_", " ").title()}  '
            f'(ratio={int(ABLATION_RATIO * 100)}%,  {config.DATASET})',
            fontsize=12, fontweight='bold'
        )
        ax.set_ylim([max(0, min(means) - 0.05), min(1.0, max(means) + 0.05)])
        ax.grid(axis='y', alpha=0.3)

        for bar, mu in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f'{mu:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    _save_fig('ablation_bar.pdf')


def plot_hyperparam_sensitivity(hp_results):
    """Two-panel plot: alpha sweep (left) and window_size sweep (right)."""
    print("  Generating hyperparameter sensitivity plot …")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, (sweep_key, sweep_values, label) in zip(
        axes,
        [('alpha',  ALPHA_VALUES,  'Alpha (α)'),
         ('window', WINDOW_SIZES,  'Window Size (W)')]
    ):
        xs, means, stds = [], [], []
        for v in sweep_values:
            key  = f'{sweep_key}_{v}'
            runs = hp_results.get(sweep_key, {}).get(key, [])
            if runs:
                mu, std = final_accuracy_stats(runs)
                xs.append(str(v))
                means.append(mu)
                stds.append(std)

        ax.bar(xs, means, color='#9b59b6', alpha=0.85,
               yerr=stds, capsize=4, edgecolor='black', linewidth=0.5)
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Mean Final Accuracy', fontsize=11)
        ax.set_title(
            f'{label} Sensitivity  ({HYPERPARAM_ATTACK.replace("_", " ").title()})',
            fontsize=11, fontweight='bold'
        )
        ax.set_ylim([max(0, min(means) - 0.05) if means else 0,
                     min(1.0, max(means) + 0.05) if means else 1.0])
        ax.grid(axis='y', alpha=0.3)
        for i, (mu, std) in enumerate(zip(means, stds)):
            ax.text(i, mu + 0.003, f'{mu:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    _save_fig('hyperparam_sensitivity.pdf')


def plot_convergence_speed(results):
    """
    Convergence round comparison: grouped bar chart.
    Lower = faster convergence.
    """
    print("  Generating convergence speed plot …")

    ratio    = 0.3
    rkey     = f'ratio_{ratio}'
    target   = 0.70   # CIFAR-10 target accuracy
    x        = np.arange(len(V2_ATTACK_TYPES))
    width    = 0.7 / len(V2_ALGORITHMS)

    fig, ax = plt.subplots(figsize=(12, 4.5))

    for i, algo in enumerate(V2_ALGORITHMS):
        means = []
        for atk in V2_ATTACK_TYPES:
            runs = results.get(algo, {}).get(atk, {}).get(rkey, [])
            if runs:
                means.append(float(np.mean([compute_convergence_round(r, target)
                                            for r in runs])))
            else:
                means.append(config.NUM_ROUNDS)

        offset = (i - len(V2_ALGORITHMS) / 2 + 0.5) * width
        ax.bar(x + offset, means, width * 0.92,
               label=algo, color=_COLORS.get(algo, 'gray'), alpha=0.85)

    ax.set_xlabel('Attack Type', fontsize=11)
    ax.set_ylabel(f'Convergence Round (first ≥ {target:.0%})', fontsize=11)
    ax.set_title(
        f'Convergence Speed  (ratio={int(ratio * 100)}%,  {config.DATASET})',
        fontsize=12, fontweight='bold'
    )
    ax.set_xticks(x)
    ax.set_xticklabels([_ATTACK_LABELS.get(a, a) for a in V2_ATTACK_TYPES], fontsize=9)
    ax.legend(fontsize=8)
    ax.set_ylim([0, config.NUM_ROUNDS + 5])
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _save_fig('convergence_speed.pdf')


def generate_all_plots(main_results, ablation_results=None, hp_results=None):
    """Generate all publication-ready PDF figures."""
    print("\nGenerating all plots …")
    plot_learning_curves(main_results)
    plot_summary_bar(main_results)
    plot_attack_phase_accuracy(main_results)
    plot_trust_evolution(main_results)
    plot_convergence_speed(main_results)
    if ablation_results:
        plot_ablation(ablation_results)
    if hp_results:
        plot_hyperparam_sensitivity(hp_results)
    print(f"  All plots saved to {config.BASE_OUTPUT_DIR}/plots/\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FedTemporal v2 — IEEE Journal Suite')
    parser.add_argument('--sanity',    action='store_true',
                        help='Run MNIST sanity check only')
    parser.add_argument('--ablation',  action='store_true',
                        help='Run ablation study only')
    parser.add_argument('--hyperparam', action='store_true',
                        help='Run hyperparameter sensitivity only')
    parser.add_argument('--baseline',   action='store_true',
                        help='Run clean baseline (no attack) only')
    parser.add_argument('--fmnist',        action='store_true',
                        help='Run Fashion-MNIST validation suite')
    parser.add_argument('--new-baselines', action='store_true',
                        help='Run new baseline algorithms only (TrimmedMean/Krum/CoordMedian/FLAME/FLDetector)')
    parser.add_argument('--merge',         action='store_true',
                        help='Merge new_baselines results into main results')
    parser.add_argument('--lie',           action='store_true',
                        help='Run focused LIE attack experiment (CIFAR-10, all algos, ratios 0.2-0.5, 5 runs)')
    parser.add_argument('--iiot',          action='store_true',
                        help='Run IIoT experiment — all 6 algorithms on one GPU (~450 runs)')
    parser.add_argument('--iiot-A',        action='store_true',
                        help='Run IIoT Part A (FedAvg/TrimmedMean/Krum) — use on GPU 0')
    parser.add_argument('--iiot-B',        action='store_true',
                        help='Run IIoT Part B (FLTrust/VFL_TS/FedTemporal) — use on GPU 1')
    parser.add_argument('--merge-iiot',    action='store_true',
                        help='Merge Part A + Part B results into fedtemporal_results_v2_iiot/')
    args = parser.parse_args()

    print("\n" + "=" * 68)
    print("  FedTemporal v2 — IEEE Journal Experimental Suite")
    print(f"  Dataset  : {config.DATASET}   Rounds: {config.NUM_ROUNDS}   "
          f"Runs: {config.NUM_RUNS}")
    print(f"  Output   : {config.BASE_OUTPUT_DIR}/")
    print("=" * 68 + "\n")

    if args.baseline:
        run_baseline()

    elif args.fmnist:
        run_fmnist_experiments()

    elif getattr(args, 'new_baselines', False):
        run_new_baselines()

    elif args.merge:
        main_path  = f'{config.BASE_OUTPUT_DIR}/data/fedtemporal_v2_main.json'
        new_path   = f'{config.BASE_OUTPUT_DIR}/data/fedtemporal_v2_new_baselines.json'
        merged     = merge_results(main_path, new_path, out_tag='merged')
        all_algos  = list(merged.keys())
        print_results_table(merged, V2_ATTACK_TYPES, V2_ATTACK_RATIOS, all_algos)

    elif args.lie:
        run_lie_experiments()

    elif args.iiot:
        run_iiot_experiments()

    elif getattr(args, 'iiot_A', False):
        run_iiot_experiments(part='A')

    elif getattr(args, 'iiot_B', False):
        run_iiot_experiments(part='B')

    elif getattr(args, 'merge_iiot', False):
        merge_iiot_parts()

    elif args.sanity:
        run_sanity_check()

    elif args.ablation:
        abl_results = run_ablation_experiments()
        plot_ablation(abl_results)

    elif args.hyperparam:
        hp_results = run_hyperparam_sensitivity()
        plot_hyperparam_sensitivity(hp_results)

    else:
        # Full suite
        main_results = run_main_experiments()
        abl_results  = run_ablation_experiments()
        hp_results   = run_hyperparam_sensitivity()

        # Plots
        generate_all_plots(main_results, abl_results, hp_results)

        # Results table
        print_results_table(main_results, V2_ATTACK_TYPES, V2_ATTACK_RATIOS,
                            V2_ALGORITHMS)

        # Statistical tests
        wtest = run_wilcoxon_tests(main_results)
        print_wilcoxon_table(wtest)

        print("\n" + "=" * 68)
        print("  DONE")
        print(f"  Data   : {config.BASE_OUTPUT_DIR}/data/")
        print(f"  Plots  : {config.BASE_OUTPUT_DIR}/plots/")
        print("=" * 68 + "\n")
