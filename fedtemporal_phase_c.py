"""
FedTemporal Phase C — GPU Experiments (JISAS-D-26-02936 revision)
=================================================================
Reviewer concerns addressed:
  R4.3 / R2.1  whitebox  — white-box attack knowing w=5, alpha=0.5, V/T/Z weights
  R2.6         trimmed_adaptive — TrimmedMean with oracle trim_rate=f
  R2.5         extraruns — 5 extra gradient-drift runs (seeds 5-9) for n=10 total

Usage:
  CUDA_VISIBLE_DEVICES=0 python fedtemporal_phase_c.py --task whitebox
  CUDA_VISIBLE_DEVICES=1 python fedtemporal_phase_c.py --task trimmed_adaptive
  CUDA_VISIBLE_DEVICES=1 python fedtemporal_phase_c.py --task extraruns --seed_offset 5

Results:
  ./fedtemporal_results_v2_whitebox/data/fedtemporal_v2_whitebox.json
  ./fedtemporal_results_v2_trimmed_adaptive/data/fedtemporal_v2_trimmed_adaptive.json
  ./fedtemporal_results_v2_extraruns/data/fedtemporal_v2_extraruns.json
"""

import os, sys, time, json, pickle, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np

# ─── CLI (must happen before TF import) ───────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--task', required=True,
                    choices=['whitebox', 'trimmed_adaptive', 'extraruns'])
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--seed_offset', type=int, default=0,
                    help='For extraruns: 5 gives seeds 5-9 (appended to existing 0-4)')
args = parser.parse_args()

if args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

# ─── TF: allow memory growth so two processes can share the GPU if needed ──────
import tensorflow as tf
for _g in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_g, True)
print(f"TF GPUs: {len(tf.config.experimental.list_physical_devices('GPU'))}  "
      f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','unset')}")

# ─── Load base code (same exec trick as fedtemporal_v2.py) ─────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
for _fpath, _tag in [('temp-main-comp-V2.py', '_base_'),
                     ('fedtemporal_v2.py',     '_ftv2_')]:
    _full = os.path.join(_HERE, _fpath)
    print(f"exec: {_full}")
    _saved = __name__
    __name__ = _tag
    with open(_full, encoding='utf-8') as _fh:
        exec(compile(_fh.read(), _full, 'exec'), globals())
    __name__ = _saved
print("Base + FedTemporal loaded.\n")

# ─── Output dir ────────────────────────────────────────────────────────────────
_OUT_DIRS = {
    'whitebox':          './fedtemporal_results_v2_whitebox',
    'trimmed_adaptive':  './fedtemporal_results_v2_trimmed_adaptive',
    'extraruns':         './fedtemporal_results_v2_extraruns',
}
OUT_DIR = _OUT_DIRS[args.task]
os.makedirs(os.path.join(OUT_DIR, 'data'),  exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'plots'), exist_ok=True)


# ============================================================================
# WHITE-BOX ATTACK IMPLEMENTATION
# ============================================================================

def _wb_flat(w, g):
    """Flatten (w - g) into 1-D array."""
    return np.concatenate([(wi - gi).flatten() for wi, gi in zip(w, g)])

def _wb_unflatten(d, g):
    """Restore 1-D array d as g + delta."""
    out, pos = [], 0
    for gi in g:
        n = gi.size
        out.append(d[pos:pos+n].reshape(gi.shape) + gi)
        pos += n
    return out

def apply_whitebox_attack(weights, global_weights, prev_global_weights,
                           round_num, run_id, warmup=15,
                           target_cos=0.30, noise_frac=0.04, base_seed=1337):
    """
    White-box adaptive attack against FedTemporal (knows w=5, alpha=0.5).

    Evasion strategy
    ----------------
    (i) Calibrated constant cosine similarity: the attacker crafts its update
        so that s_i(t) = target_cos for every attack round.
        A constant s_i sequence has V=0, T=0, Z=0 — all three temporal signals
        are suppressed simultaneously.

    (ii) Poison in orthogonal subspace: the poisoning energy is placed in the
         direction ORTHOGONAL to the server gradient, maximising accuracy damage
         while maintaining the desired s_i = target_cos.

    (iii) Calibrated noise: small random perturbation (noise_frac * ||d||)
          prevents perfect V=0, keeping the attacker's variance within the
          normal non-IID client population range and avoiding statistical
          detection of zero-variance anomaly.

    (iv) 15-round honest warm-up: shorter than sleeper mimicry (15 vs 30)
         to be more aggressive while still building initial trust.

    Note: even with V=T=Z=0, the instant trust s_i = 0.30 is below honest
    clients' typical s_i ~ 0.85, so FedTemporal still partially downweights
    malicious clients. The experimental results will quantify this residual
    detection capability.
    """
    if prev_global_weights is None or round_num < warmup:
        return weights   # honest warmup or no history yet

    # Server gradient direction (θ(t) − θ(t-1))
    d_server = _wb_flat(global_weights, prev_global_weights)
    norm_srv  = np.linalg.norm(d_server)
    if norm_srv < 1e-10:
        return weights
    d_srv_hat = d_server / norm_srv

    # Client honest update
    d_honest   = _wb_flat(weights, global_weights)
    honest_norm = np.linalg.norm(d_honest)
    if honest_norm < 1e-10:
        return weights

    # Poison direction: sign-flip projected orthogonal to server direction
    w_flip  = apply_byzantine_attack(weights, global_weights, 'signflip', 1.0)
    d_flip  = _wb_flat(w_flip, global_weights)
    d_flip_orth = d_flip - np.dot(d_flip, d_srv_hat) * d_srv_hat
    norm_orth   = np.linalg.norm(d_flip_orth)
    if norm_orth < 1e-10:
        d_flip_orth = -d_srv_hat.copy(); norm_orth = 1.0
    d_orth_hat  = d_flip_orth / norm_orth

    # Combine: target_cos in server dir + sin(acos(c)) in orthogonal poison dir
    c = float(target_cos)
    s = float(np.sqrt(max(0.0, 1.0 - c * c)))
    d_attack = c * d_srv_hat + s * d_orth_hat

    # Calibrated noise (reproducible: seeded by run+round)
    rng   = np.random.RandomState(base_seed + run_id * 10000 + round_num * 37)
    noise = rng.randn(len(d_attack)) * noise_frac
    d_attack += noise

    # Rescale to honest gradient magnitude
    atk_norm = np.linalg.norm(d_attack)
    if atk_norm > 1e-10:
        d_attack *= (honest_norm / atk_norm)

    return _wb_unflatten(d_attack, global_weights)


# ============================================================================
# CUSTOM TRAINING LOOP FOR WHITEBOX ATTACK
# (mirrors run_federated_training_v3 exactly, adds whitebox injection)
# ============================================================================

def run_whitebox_training(algorithm, client_data, X_test, y_test,
                           attack_ratio=0.0, run_id=0):
    """
    Training loop for white-box adaptive attack. Mirrors run_federated_training_v3.
    Attack is injected after honest training for malicious clients.
    """
    n_clients = len(client_data)
    n_mal     = int(n_clients * attack_ratio)
    rng_sel   = np.random.RandomState(config.RANDOM_SEED + run_id)
    malicious = set(rng_sel.choice(n_clients, n_mal, replace=False).tolist()) \
                if n_mal > 0 else set()

    if hasattr(algorithm, 'initialize'):
        algorithm.initialize(n_clients)
    else:
        algorithm.initialize_model()

    if isinstance(algorithm, VFLTSAlgorithm):
        algorithm.malicious_clients = malicious

    print(f"\n{'='*68}")
    print(f"  Algorithm : {algorithm.name}")
    print(f"  Attack    : whitebox_adaptive   Ratio: {attack_ratio}   Run: {run_id+1}")
    print(f"  Warmup    : 15 rounds  (attack starts round 16)")
    print(f"  Malicious : {sorted(malicious)}")
    print(f"{'='*68}\n")

    history = {
        'accuracy':         [],
        'loss':             [],
        'training_time':    [],
        'malicious_clients': sorted(malicious),
        'trust_scores':     [],
        'attack_active':    [],
    }

    prev_global = None

    for rnd in range(config.NUM_ROUNDS):
        t0          = time.time()
        cur_global  = [w.copy() for w in algorithm.global_model.get_weights()]
        attack_active = (rnd >= 15)

        k        = max(1, int(n_clients * config.CLIENT_PARTICIPATION_RATE))
        selected = algorithm.select_clients(rnd, client_data, k)

        local_weights     = []
        validation_losses = []

        for cid in selected:
            is_mal = cid in malicious

            if isinstance(algorithm, VFLTSAlgorithm):
                w, _, vl = algorithm.train_client(
                    cid, client_data[cid], cur_global, None, False)
                validation_losses.append(vl)
            else:
                w, _ = algorithm.train_client(
                    cid, client_data[cid], cur_global, None, False)

            if w is None:
                continue

            # Inject white-box attack for malicious clients
            if is_mal and attack_active:
                w = apply_whitebox_attack(
                    w, cur_global, prev_global, rnd, run_id)

            local_weights.append(w)

        if not local_weights:
            history['attack_active'].append(attack_active)
            history['trust_scores'].append({})
            continue

        # Aggregation — capture return value and apply to global model
        if isinstance(algorithm, VFLTSAlgorithm):
            agg = algorithm.aggregate(local_weights, selected,
                                      validation_losses, client_data, rnd)
        else:
            agg = algorithm.aggregate(local_weights, selected, client_data)

        if agg:
            algorithm.global_model.set_weights(agg)

        prev_global = cur_global   # θ(t-1) for next round

        # Evaluation — use global model directly (matches run_federated_training_v3)
        test_loss, test_acc = algorithm.global_model.evaluate(X_test, y_test, verbose=0)
        acc, los = test_acc, test_loss

        elapsed = time.time() - t0
        history['accuracy'].append(float(acc))
        history['loss'].append(float(los))
        history['training_time'].append(float(elapsed))
        history['attack_active'].append(attack_active)

        # Trust scores (FedTemporal only)
        ts = ({str(k): float(v) for k, v in algorithm.trust_scores.items()}
              if hasattr(algorithm, 'trust_scores') else {})
        history['trust_scores'].append(ts)

        if config.VERBOSE and (rnd + 1) % 10 == 0:
            phase = '[WB-ATK]' if attack_active else '[warmup]'
            print(f"  Round {rnd+1:>3}/{config.NUM_ROUNDS} {phase}  "
                  f"acc={acc:.4f}  t={elapsed:.1f}s")

    return history


# ============================================================================
# ADAPTIVE TRIMMED MEAN
# ============================================================================

class AdaptiveTrimmedMeanAlgorithm(TrimmedMeanAlgorithm):
    """
    Oracle TrimmedMean: trim_ratio = Byzantine ratio f.
    Theoretically optimal: trims exactly the Byzantine clients from each tail,
    wasting the minimum number of honest clients.
    Standard TrimmedMean uses a fixed 20% trim regardless of the actual f.
    """
    def __init__(self, input_shape, num_classes, attack_ratio=0.3):
        super().__init__(input_shape, num_classes, trim_ratio=attack_ratio)
        self.name = 'TrimmedMean_Adaptive'


# ============================================================================
# SAVE
# ============================================================================

def _to_json_safe(obj):
    """Recursively convert numpy types to Python natives for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    return obj

def save_results_c(results, tag):
    json_path = os.path.join(OUT_DIR, 'data', f'fedtemporal_v2_{tag}.json')
    pkl_path  = os.path.join(OUT_DIR, 'data', f'fedtemporal_v2_{tag}.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f, protocol=4)
    with open(json_path, 'w') as f:
        json.dump(_to_json_safe(results), f, indent=2)
    print(f"  [Saved -> {json_path}]")


# ============================================================================
# TASK: WHITEBOX
# ============================================================================

def task_whitebox():
    """
    White-box adaptive attack vs FedAvg, FLTrust, VFL_TS, FedTemporal.
    4 algorithms × 4 ratios × 5 runs = 80 runs.
    """
    (X_test, y_test), n_cls, shape, client_data, shared_val, root_dataset = \
        _load_data_and_roots()

    algorithms = ['FedAvg', 'FLTrust', 'VFL_TS', 'FedTemporal']
    ratios     = [0.2, 0.3, 0.4, 0.5]
    num_runs   = args.runs
    tag        = 'whitebox'

    total = len(algorithms) * len(ratios) * num_runs
    count = 0
    results = {}

    for algo_name in algorithms:
        results[algo_name] = {'whitebox_adaptive': {}}
        for ratio in ratios:
            rkey = f'ratio_{ratio}'
            results[algo_name]['whitebox_adaptive'][rkey] = []
            for run_id in range(num_runs):
                count += 1
                print(f"\n[{count}/{total}] {algo_name} | whitebox_adaptive | "
                      f"ratio={ratio} | run={run_id+1}")
                np.random.seed(config.RANDOM_SEED + run_id)
                tf.random.set_seed(config.RANDOM_SEED + run_id)
                algo = make_algorithm(algo_name, shape, n_cls,
                                      root_dataset, shared_val)
                hist = run_whitebox_training(
                    algo, client_data, X_test, y_test,
                    attack_ratio=ratio, run_id=run_id)
                results[algo_name]['whitebox_adaptive'][rkey].append(hist)

        save_results_c(results, tag)   # crash-safe: save after each algorithm
    return results


# ============================================================================
# TASK: TRIMMED MEAN ADAPTIVE
# ============================================================================

def task_trimmed_adaptive():
    """
    Oracle TrimmedMean (trim_rate = f) on all 4 main attacks.
    1 algorithm × 4 attacks × 4 ratios × 5 runs = 80 runs.
    """
    (X_test, y_test), n_cls, shape, client_data, shared_val, root_dataset = \
        _load_data_and_roots()

    attack_types = ['byzantine_signflip', 'sleeper_mimicry',
                    'oscillating_strategic', 'gradient_drift']
    ratios   = [0.2, 0.3, 0.4, 0.5]
    num_runs = args.runs
    tag      = 'trimmed_adaptive'

    adaptive_params = {
        'k_warmup'        : max(1, int(config.NUM_ROUNDS * SLEEPER_WARMUP_FRACTION)),
        'period'          : OSCILLATING_PERIOD_V2,
        'poison_strength' : MIMICRY_POISON_STRENGTH,
        'drift_rate'      : GRADIENT_DRIFT_RATE,
        'base_attack'     : 'signflip',
        'amplification'   : 3.0,
        'lie_amplification': LIE_AMPLIFICATION,
    }

    # Resume from existing PKL if present (crash recovery)
    pkl_path = os.path.join(OUT_DIR, 'data', 'fedtemporal_v2_trimmed_adaptive.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as _f:
            results = pickle.load(_f)
        done = set(results.get('TrimmedMean_Adaptive', {}).keys())
        attack_types = [a for a in attack_types if a not in done]
        print(f"  [RESUMED] Already done: {done}  Remaining: {attack_types}")
    else:
        results = {'TrimmedMean_Adaptive': {}}

    total = len(attack_types) * len(ratios) * num_runs
    count = 0

    for atk in attack_types:
        results['TrimmedMean_Adaptive'][atk] = {}
        for ratio in ratios:
            rkey = f'ratio_{ratio}'
            results['TrimmedMean_Adaptive'][atk][rkey] = []
            for run_id in range(num_runs):
                count += 1
                print(f"\n[{count}/{total}] TrimmedMean_Adaptive | {atk} | "
                      f"ratio={ratio} | run={run_id+1}")
                np.random.seed(config.RANDOM_SEED + run_id)
                tf.random.set_seed(config.RANDOM_SEED + run_id)
                algo = AdaptiveTrimmedMeanAlgorithm(shape, n_cls,
                                                     attack_ratio=ratio)
                hist = run_federated_training_v3(
                    algo, client_data, X_test, y_test,
                    attack_type=atk, attack_ratio=ratio, run_id=run_id,
                    adaptive_params=adaptive_params)
                results['TrimmedMean_Adaptive'][atk][rkey].append(hist)

        save_results_c(results, tag)
    return results


# ============================================================================
# TASK: EXTRA RUNS (gradient_drift, seeds 5-9)
# ============================================================================

def task_extraruns():
    """
    5 extra runs for gradient_drift: FedTemporal + FLTrust.
    Combined with existing 5 (seeds 0-4) gives n=10 for Wilcoxon at p=0.020.
    """
    (X_test, y_test), n_cls, shape, client_data, shared_val, root_dataset = \
        _load_data_and_roots()

    algorithms  = ['FedTemporal', 'FLTrust']
    ratios      = [0.2, 0.3, 0.4, 0.5]
    num_runs    = args.runs
    seed_offset = args.seed_offset
    tag         = 'extraruns'

    adaptive_params = {
        'k_warmup'        : max(1, int(config.NUM_ROUNDS * SLEEPER_WARMUP_FRACTION)),
        'period'          : OSCILLATING_PERIOD_V2,
        'poison_strength' : MIMICRY_POISON_STRENGTH,
        'drift_rate'      : GRADIENT_DRIFT_RATE,
        'base_attack'     : 'signflip',
        'amplification'   : 3.0,
        'lie_amplification': LIE_AMPLIFICATION,
    }

    total = len(algorithms) * len(ratios) * num_runs
    count = 0
    results = {a: {'gradient_drift': {}} for a in algorithms}

    for algo_name in algorithms:
        for ratio in ratios:
            rkey = f'ratio_{ratio}'
            results[algo_name]['gradient_drift'][rkey] = []
            for run_idx in range(num_runs):
                run_id = run_idx + seed_offset
                count += 1
                print(f"\n[{count}/{total}] {algo_name} | gradient_drift | "
                      f"ratio={ratio} | run={run_id+1} (seed={config.RANDOM_SEED+run_id})")
                np.random.seed(config.RANDOM_SEED + run_id)
                tf.random.set_seed(config.RANDOM_SEED + run_id)
                algo = make_algorithm(algo_name, shape, n_cls,
                                      root_dataset, shared_val)
                hist = run_federated_training_v3(
                    algo, client_data, X_test, y_test,
                    attack_type='gradient_drift', attack_ratio=ratio,
                    run_id=run_id, adaptive_params=adaptive_params)
                results[algo_name]['gradient_drift'][rkey].append(hist)

        save_results_c(results, tag)
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"\n{'='*62}")
    print(f"FedTemporal Phase C  |  Task: {args.task}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','unset')}")
    print(f"Output: {OUT_DIR}  |  Runs: {args.runs}")
    print(f"{'='*62}\n")

    t0 = time.time()
    if   args.task == 'whitebox':          task_whitebox()
    elif args.task == 'trimmed_adaptive':  task_trimmed_adaptive()
    elif args.task == 'extraruns':         task_extraruns()
    elapsed = time.time() - t0

    print(f"\n{'='*62}")
    print(f"Task '{args.task}' done. Elapsed: {elapsed/3600:.2f}h")
    print(f"Results: {OUT_DIR}/data/")
    print(f"{'='*62}")

if __name__ == '__main__':
    main()
