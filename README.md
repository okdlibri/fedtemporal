# FedTemporal: Byzantine-Robust Federated Learning via Temporal Behavioral Profiling

**FedTemporal** is a Byzantine-robust federated learning algorithm that detects adaptive adversaries by profiling the *temporal behavior* of clients across rounds, rather than evaluating each round in isolation. It is specifically designed to counter attacks that evade per-round defenses by mimicking benign gradients in early rounds or drifting gradually over time.

---

## Paper

> **FedTemporal: Temporal Behavioral Profiling for Byzantine-Robust Federated Learning**  
> Omar Dib  
> *Under review at the IEEE Journal of Selected Areas in Communications (JISAS)*

If you use this code, please cite:

```bibtex
@misc{dib2026fedtemporal,
  author    = {Dib, Omar},
  title     = {{FedTemporal}: Temporal Behavioral Profiling for Byzantine-Robust Federated Learning},
  year      = {2026},
  note      = {Under review at IEEE JISAS}
}
```

This work extends and compares against:

```bibtex
@misc{dib2025vflts,
  author    = {Dib, Omar},
  title     = {Practical Byzantine-Resilient Federated Learning via Zero-Knowledge Proofs and Adaptive Reputation},
  year      = {2026},
  doi       = {10.5281/zenodo.23084101},
  url       = {https://doi.org/10.5281/zenodo.23084101},
  note      = {Preprint. Submitted to Knowledge-Based Systems (Elsevier)}
}
```

---

## Algorithm Overview

FedTemporal computes a **temporal anomaly score** for each client by combining three signals over a sliding window of past rounds:

| Signal | Weight | Description |
|--------|--------|-------------|
| Variance anomaly (V) | 0.40 | Unusual variability in gradient updates |
| Trend anomaly (T) | 0.30 | Monotonic drift in client behavior |
| Z-score anomaly (Z) | 0.30 | Statistical outlier detection across clients |

The trust score for each client uses **exponential moving average (EMA) inertia**:

```
trust(t) = alpha * (instant_trust * temporal_score) + (1 - alpha) * trust(t-1)
```

Default parameters: `window_size=5`, `alpha=0.5`.

---

## Repository Structure

```
.
├── temp-main-comp-V2.py       # Baseline implementations (FedAvg, FLTrust, VFL-TS)
├── fedtemporal_v2.py          # FedTemporal algorithm + full experimental suite
├── fedtemporal_phase_c.py     # Phase C experiments (whitebox attack, TrimmedMean oracle)
├── fedtemporal_sensitivity.py # Root dataset sensitivity analysis
└── requirements.txt
```

`fedtemporal_v2.py` and `fedtemporal_phase_c.py` load `temp-main-comp-V2.py` at runtime via `exec()` to reuse the baseline implementations without code duplication.

---

## Requirements

- Python 3.8+
- TensorFlow 2.10.1 (GPU recommended)
- See `requirements.txt` for full dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Main experimental suite (CIFAR-10, 50 rounds, 5 runs)

```bash
python fedtemporal_v2.py
```

### Quick sanity check (MNIST, 15 rounds, 3 runs)

```bash
python fedtemporal_v2.py --sanity
```

### Ablation study only

```bash
python fedtemporal_v2.py --ablation
```

### Hyperparameter sensitivity

```bash
python fedtemporal_v2.py --hyperparam
```

### Phase C experiments (reviewer additions)

```bash
# White-box adaptive attack (adversary knows w=5, alpha=0.5, V/T/Z weights)
python fedtemporal_phase_c.py --task whitebox

# TrimmedMean with oracle trim_rate = attack_ratio
python fedtemporal_phase_c.py --task trimmed_adaptive
```

### Root dataset sensitivity analysis

```bash
python fedtemporal_sensitivity.py
```

---

## Experimental Setup

| Setting | Value |
|---------|-------|
| Dataset | CIFAR-10 |
| Clients | 20 |
| Participation | 80% per round |
| Rounds | 50 |
| Local epochs | 4 |
| Optimizer | Adam (lr=0.001) |
| Non-IID split | Dirichlet α=0.5 |
| Attack ratios | 0.2, 0.3, 0.4, 0.5 |

### Attack Suite

| Attack | Mechanism |
|--------|-----------|
| `byzantine_signflip` | Static sign-flip (reference) |
| `sleeper_mimicry` | 60% honest warmup → black-box gradient mimicry |
| `oscillating_strategic` | Period-4 alternation (2 attack + 2 honest rounds) |
| `gradient_drift` | Gradual directional shift from honest to signflip over 30 rounds |

### Baselines

- **FedAvg** — no defense
- **FLTrust** — root dataset cosine similarity trust scores
- **VFL-TS** — reputation-based aggregation with validation loss feedback
- **TrimmedMean** (oracle) — per-round coordinate trimming with known attack ratio

---

## Key Results

On CIFAR-10 at attack ratio=0.5 under gradient drift:

| Method | Accuracy |
|--------|----------|
| FedAvg | 0.675 ± 0.009 |
| FLTrust | 0.619 ± 0.015 |
| VFL-TS | 0.348 ± 0.088 |
| **FedTemporal** | **0.625 ± 0.006** |

FedTemporal maintains accuracy where VFL-TS collapses (Δ=0.277) because per-round reputation scores cannot detect gradual behavioral drift. FedTemporal's temporal window captures the cumulative pattern.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

Omar Dib — odib@kean.edu
