"""
FedTemporal Phase C — Sensitivity Analysis (R2.4)
==================================================
Tests robustness of FedTemporal to root dataset perturbation.
Three conditions × gradient_drift ratio=0.3 × 3 runs each = 9 runs.

Conditions:
  root_small  : 50 samples (5/class) instead of 100
  root_large  : 200 samples (20/class)
  root_imbal  : 100 samples, skewed (5 over-represented, 5 under-represented)
  root_contam : 100 samples, 10% sign-flipped (Byzantine-perturbed)

Baseline (100 samples, balanced) already known from main results.
Expected: <3% accuracy variation across conditions.

Usage:
  CUDA_VISIBLE_DEVICES=0 python fedtemporal_sensitivity.py
"""

import os, sys, time, json, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import tensorflow as tf
for _g in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_g, True)
print(f"TF GPUs: {len(tf.config.experimental.list_physical_devices('GPU'))}  "
      f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','unset')}")

# ── Load base code ─────────────────────────────────────────────────────────────
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

OUT_DIR = './fedtemporal_results_v2_sensitivity'
os.makedirs(os.path.join(OUT_DIR, 'data'), exist_ok=True)


# ── JSON serialisation helper ──────────────────────────────────────────────────
def _to_json_safe(obj):
    if isinstance(obj, np.ndarray):      return obj.tolist()
    if isinstance(obj, np.integer):      return int(obj)
    if isinstance(obj, np.floating):     return float(obj)
    if isinstance(obj, dict):            return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):   return [_to_json_safe(x) for x in obj]
    return obj


def save_results(results):
    pkl  = os.path.join(OUT_DIR, 'data', 'fedtemporal_v2_sensitivity.pkl')
    js   = os.path.join(OUT_DIR, 'data', 'fedtemporal_v2_sensitivity.json')
    with open(pkl, 'wb') as f:
        pickle.dump(results, f, protocol=4)
    with open(js, 'w') as f:
        json.dump(_to_json_safe(results), f, indent=2)
    print(f"  [Saved -> {js}]")


# ── Root dataset builders ──────────────────────────────────────────────────────
def _build_root_variants(Xsv, ysv, n_classes=10):
    """Return dict of (X, y) tuples for each sensitivity condition."""
    variants = {}
    classes = np.unique(ysv)

    # ── small: 50 samples, ~5/class ──────────────────────────────────────────
    idx_small = []
    per_class = max(1, 50 // n_classes)
    for c in classes:
        ci = np.where(ysv == c)[0][:per_class]
        idx_small.extend(ci.tolist())
    idx_small = np.array(idx_small[:50])
    variants['root_small'] = (Xsv[idx_small], ysv[idx_small])

    # ── large: 200 samples, ~20/class ────────────────────────────────────────
    idx_large = []
    per_class = max(1, 200 // n_classes)
    for c in classes:
        ci = np.where(ysv == c)[0][:per_class]
        idx_large.extend(ci.tolist())
    idx_large = np.array(idx_large[:200])
    variants['root_large'] = (Xsv[idx_large], ysv[idx_large])

    # ── imbalanced: 100 total, 5 classes × 15 samples, 5 classes × 5 samples ─
    idx_imbal = []
    for i, c in enumerate(classes):
        ci = np.where(ysv == c)[0]
        n  = 15 if i < 5 else 5
        idx_imbal.extend(ci[:n].tolist())
    variants['root_imbal'] = (Xsv[idx_imbal], ysv[idx_imbal])

    # ── contaminated: 100 samples, 10 sign-flipped ───────────────────────────
    idx_cont = []
    per_class = max(1, 100 // n_classes)
    for c in classes:
        ci = np.where(ysv == c)[0][:per_class]
        idx_cont.extend(ci.tolist())
    Xcont = Xsv[idx_cont].copy()
    ycont = ysv[idx_cont].copy()
    n_poison = max(1, int(len(Xcont) * 0.10))
    Xcont[:n_poison] = -Xcont[:n_poison]                # sign-flip
    Xcont = np.clip(Xcont, 0.0, 1.0)
    variants['root_contam'] = (Xcont, ycont)

    return variants


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    # Load data (same as phase_c.py)
    (X_train, y_train), (X_test, y_test), n_cls, shape = load_dataset(config.DATASET)
    client_data = create_non_iid_split(
        X_train, y_train, config.NUM_CLIENTS, config.NON_IID_ALPHA
    )
    shared_val = construct_shared_validation_set(
        client_data,
        config.SHARED_VAL_SAMPLES_PER_CLIENT,
        config.SHARED_VAL_MAX_SIZE
    )
    Xsv, ysv = shared_val

    # Standard root (baseline reference — not re-run here, already in main results)
    root_n = max(100, int(len(Xsv) * 0.3))

    root_variants = _build_root_variants(Xsv[:root_n], ysv[:root_n], n_classes=n_cls)
    for cond, (Xr, yr) in root_variants.items():
        print(f"  Root variant '{cond}': n={len(Xr)}  "
              f"class_counts={np.bincount(yr.astype(int)).tolist()}")

    # Adaptive params — same as phase_c extraruns
    adaptive_params = {
        'k_warmup'         : max(1, int(config.NUM_ROUNDS * SLEEPER_WARMUP_FRACTION)),
        'period'           : OSCILLATING_PERIOD_V2,
        'poison_strength'  : MIMICRY_POISON_STRENGTH,
        'drift_rate'       : GRADIENT_DRIFT_RATE,
        'base_attack'      : 'signflip',
        'amplification'    : 3.0,
        'lie_amplification': LIE_AMPLIFICATION,
    }

    RATIO   = 0.3
    N_RUNS  = 3
    results = {}

    conditions  = list(root_variants.keys())
    total_runs  = len(conditions) * N_RUNS
    count       = 0

    for cond in conditions:
        root_dataset = root_variants[cond]
        results[cond] = {'gradient_drift': {f'ratio_{RATIO}': []}}

        for run_id in range(N_RUNS):
            count += 1
            print(f"\n[{count}/{total_runs}] FedTemporal | gradient_drift | "
                  f"ratio={RATIO} | {cond} | run={run_id+1}")
            np.random.seed(config.RANDOM_SEED + run_id)
            tf.random.set_seed(config.RANDOM_SEED + run_id)

            algo = make_algorithm('FedTemporal', shape, n_cls,
                                  root_dataset, shared_val)
            hist = run_federated_training_v3(
                algo, client_data, X_test, y_test,
                attack_type='gradient_drift', attack_ratio=RATIO,
                run_id=run_id, adaptive_params=adaptive_params)

            results[cond]['gradient_drift'][f'ratio_{RATIO}'].append(hist)

        # Save after each condition
        save_results(results)

    # Print summary
    print('\n' + '='*60)
    print(f'Sensitivity Analysis — FedTemporal | gradient_drift | ratio={RATIO}')
    print(f'Baseline (n=5): 0.637±0.014  (from main results)')
    print('='*60)
    for cond in conditions:
        runs = results[cond]['gradient_drift'][f'ratio_{RATIO}']
        accs = [r['accuracy'][-1] for r in runs if r.get('accuracy')]
        m, s = np.mean(accs), np.std(accs)
        print(f'  {cond:<18}: {m:.4f}±{s:.4f}  '
              f'runs={[round(a,3) for a in accs]}  '
              f'Δ vs baseline={m-0.637:+.4f}')

    print(f'\nTotal time: {(time.time()-t0)/3600:.2f}h')
    print(f'Results: {OUT_DIR}/data/')


if __name__ == '__main__':
    main()
