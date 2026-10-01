"""
VFL-TS: Verifiable Federated Learning with Two-Stage Selection
COMPREHENSIVE DEBUG VERSION - Byzantine Attack Investigation

This implementation includes:
1. Multi-dimensional signal computation (6 signals per client)
2. Reputation-aware probabilistic client selection
3. Shared validation set construction
4. Robust filtering (Krum-style on validation losses)
5. Comprehensive attack simulations (Byzantine, Label-flipping, Gradient_reversal)
6. Extensive experimental framework with automated plotting and PDF export
7. Statistical validation and ablation studies
8. ✅ FIXED: UTF-8 encoding for Windows
9. ✅ FIXED: Full reputation tracking implementation
10. ✅ FIXED: Reputation evolution plotting

🔧 NEW IN THIS VERSION - COMPREHENSIVE DEBUGGING:
11. ✅ DebugTracker class for round-by-round analysis
12. ✅ Byzantine attack verification
13. ✅ Detailed Krum filtering inspection
14. ✅ Reputation evolution tracking with malicious vs honest comparison
15. ✅ False positive rate analysis
16. ✅ Attack effectiveness metrics

Author: Research Team
Date: December 2024
Target: Top-tier AI/Security Conference Publication
"""

import numpy as np
import tensorflow as tf
import pprint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving PDFs
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from sklearn.datasets import make_classification
import time
import warnings
import json
import os
from datetime import datetime
from collections import defaultdict
import pickle

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE PARAMETERS FOR YOUR EXPERIMENTS
# ============================================================================

class BaseConfig:
    """
    Base configuration with common parameters.
    Dataset-specific configs inherit from this.
    """
    # ===== Common Parameters =====
    NUM_RUNS = 5  # Number of independent runs for statistical validation
    RANDOM_SEED = 42  # Base random seed

    # ===== Output Configuration =====
    BASE_OUTPUT_DIR = './vflts_results'  # Base directory for all outputs
    OUTPUT_DIR = './vflts_results'  # Will be set dynamically per run
    DATASET = 'UNKNOWN'  # Dataset name (set by specific config)
    SAVE_PLOTS = True  # Save all plots as PDFs
    SAVE_RAW_DATA = True  # Save raw experimental data as pickle
    VERBOSE = True  # Print detailed progress
    DEBUG_MODE = True  # Print extensive debugging information

    # ===== Computational Optimization =====
    USE_GPU = True  # Use GPU if available
    MIXED_PRECISION = False  # Use mixed precision training for speed
    PARALLEL_CLIENTS = False  # Simulate clients in parallel (requires more memory)

    # ===== Attack Configurations =====
    #'None', 'byzantine_signflip', 'byzantine_gaussian', 'byzantine_scaling', 'label_flipping'
    ATTACK_TYPES = [
        'byzantine_signflip', 
        'byzantine_gaussian', 
        'byzantine_scaling', 
        'label_flipping'
    ]
    ATTACK_RATIOS = [0.2, 0.3, 0.4]  #0.2, 0.3, 0.4 Fraction of malicious clients

    # ===== Baseline Algorithms =====
    #'VFL_TS', 'FedAvg', 'FedProx', 'TrimmedMean', 'Krum', 'FLTrust'
    ALGORITHMS = [
        'VFL_TS', 'FedAvg', 'FedProx', 'TrimmedMean', 'Krum', 'FLTrust'
    ]

    # ===== Ablation Studies =====
    ABLATION_CONFIGS = [
        None,  # Full VFL-TS
    ]


class SyntheticConfig(BaseConfig):
    """Configuration optimized for SYNTHETIC dataset (fast prototyping)"""

    DATASET = 'SYNTHETIC'

    # ===== Federated Learning Setup =====
    NUM_CLIENTS = 20  # Smaller for faster experiments
    NUM_ROUNDS = 15  # Fewer rounds for quick validation
    EPOCHS_LOCAL = 3  # Quick local training
    BATCH_SIZE = 64  # Smaller batches for synthetic data
    CLIENT_PARTICIPATION_RATE = 0.7

    # ===== Non-IID Data Distribution =====
    NON_IID_ALPHA = 0.5  # Moderate non-IID
    USE_NON_IID = True

    # ===== VFL-TS Protocol Parameters =====
    REPUTATION_LAMBDA = 0.85
    REPUTATION_PENALTY = 0.25

    SELECTION_ALPHA = 0.6
    STALENESS_BETA = 0.1

    QUALITY_WEIGHT_LOSS = 1.0
    QUALITY_WEIGHT_DATA = 0.5
    QUALITY_WEIGHT_ALIGNMENT = 0.3

    KRUM_NEIGHBORS = 6  # Adjusted for 20 clients
    TRIM_RATIO = 0.2

    # Shared validation set
    SHARED_VAL_SAMPLES_PER_CLIENT = 10
    SHARED_VAL_MAX_SIZE = 300


class MNISTConfig(BaseConfig):
    """Configuration optimized for MNIST dataset"""

    DATASET = 'MNIST'

    # ===== Federated Learning Setup =====
    NUM_CLIENTS = 20
    NUM_ROUNDS = 15
    EPOCHS_LOCAL = 2
    BATCH_SIZE = 64
    CLIENT_PARTICIPATION_RATE = 0.8

    # ===== Non-IID Data Distribution =====
    NON_IID_ALPHA = 0.5  # Moderate non-IID for MNIST
    USE_NON_IID = True

    # ===== VFL-TS Protocol Parameters =====
    REPUTATION_LAMBDA = 0.95
    REPUTATION_PENALTY = 0.20

    SELECTION_ALPHA = 0.6
    STALENESS_BETA = 0.1

    QUALITY_WEIGHT_LOSS = 1.5
    QUALITY_WEIGHT_DATA = 0.3
    QUALITY_WEIGHT_ALIGNMENT = 0.2

    KRUM_NEIGHBORS = 5
    TRIM_RATIO = 0.35

    # Shared validation set
    SHARED_VAL_SAMPLES_PER_CLIENT = 10
    SHARED_VAL_MAX_SIZE = 500


class FashionMNISTConfig(BaseConfig):
    """Configuration optimized for Fashion-MNIST dataset"""

    DATASET = 'FASHION_MNIST'

    # ===== Federated Learning Setup =====
    NUM_CLIENTS = 30#30
    NUM_ROUNDS = 30#30  # More rounds for harder dataset
    EPOCHS_LOCAL = 3#3
    BATCH_SIZE = 128
    CLIENT_PARTICIPATION_RATE = 0.7

    # ===== Non-IID Data Distribution =====
    NON_IID_ALPHA = 0.5  # Moderate non-IID
    USE_NON_IID = True

    # ===== VFL-TS Protocol Parameters =====
    REPUTATION_LAMBDA = 0.95  # More aggressive for harder dataset
    REPUTATION_PENALTY = 0.20

    SELECTION_ALPHA = 0.6
    STALENESS_BETA = 0.1

    QUALITY_WEIGHT_LOSS = 1.5
    QUALITY_WEIGHT_DATA = 0.3
    QUALITY_WEIGHT_ALIGNMENT = 0.2

    KRUM_NEIGHBORS = 8
    TRIM_RATIO = 0.25

    # Shared validation set
    SHARED_VAL_SAMPLES_PER_CLIENT = 10
    SHARED_VAL_MAX_SIZE = 500


class CIFAR10Config(BaseConfig):
    """Configuration optimized for CIFAR-10 dataset"""

    DATASET = 'CIFAR10'

    # ===== Federated Learning Setup =====
    NUM_CLIENTS = 20  # More clients for larger dataset
    NUM_ROUNDS = 40  # More rounds for complex dataset
    EPOCHS_LOCAL = 4  # More local epochs for images
    BATCH_SIZE = 64  # Smaller batch for memory efficiency
    CLIENT_PARTICIPATION_RATE = 0.8

    # ===== Non-IID Data Distribution =====
    NON_IID_ALPHA = 0.5  # More extreme non-IID for CIFAR-10
    USE_NON_IID = True

    # ===== VFL-TS Protocol Parameters =====
    REPUTATION_LAMBDA = 0.95  # More conservative for complex data
    REPUTATION_PENALTY = 0.22  # Stronger penalty for attacks

    SELECTION_ALPHA = 0.6  # Higher weight on quality
    STALENESS_BETA = 0.10  # Higher staleness penalty

    QUALITY_WEIGHT_LOSS = 1.5
    QUALITY_WEIGHT_DATA = 0.3  # Higher weight on data size
    QUALITY_WEIGHT_ALIGNMENT = 0.2  # Higher weight on alignment

    KRUM_NEIGHBORS = 6  # More neighbors for 40 clients
    TRIM_RATIO = 0.3  # More aggressive trimming

    # Shared validation set
    SHARED_VAL_SAMPLES_PER_CLIENT = 20
    SHARED_VAL_MAX_SIZE = 800


# ============================================================================
# SELECT YOUR DATASET CONFIGURATION HERE
# ============================================================================
# Uncomment the config you want to use:

#config = FashionMNISTConfig()  # Current selection
#config = SyntheticConfig()
#config = MNISTConfig()
config = CIFAR10Config()

# Print configuration info
print(f"\n{'='*80}")
print(f"CONFIGURATION")
print(f"{'='*80}")
print(f"Dataset: {config.DATASET}")
print(f"Algorithms: {config.ALGORITHMS}")
print(f"Number of runs: {config.NUM_RUNS}")
print(f"Number of rounds: {config.NUM_ROUNDS}")
print(f"Attack types: {config.ATTACK_TYPES}")
print(f"Attack ratios: {config.ATTACK_RATIOS}")
print(f"Output directory: {config.BASE_OUTPUT_DIR}")
print(f"{'='*80}\n")

# Create base output directory (per-run directories will be created in experiment loop)
os.makedirs(config.BASE_OUTPUT_DIR, exist_ok=True)

# ============================================================================
# GPU CONFIGURATION
# ============================================================================

if config.USE_GPU:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU Available: {len(gpus)} device(s)")
        except RuntimeError as e:
            print(f"⚠️  GPU configuration error: {e}")
    else:
        print("⚠️  No GPU found, using CPU")
else:
    print("ℹ️  Using CPU (GPU disabled in config)")

# Set style for publication-quality plots
plt.style.use('seaborn-paper')
sns.set_palette('husl')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

# ============================================================================
# DEBUGGING UTILITIES
# ============================================================================

class DebugTracker:
    """Track detailed information for debugging Byzantine attacks."""
    
    def __init__(self):
        self.round_data = []
        
    def log_round(self, round_num, selected_clients, malicious_clients,
                  client_updates, validation_losses, reputations,
                  kept_after_filtering):
        """Log detailed information for each round."""
        
        # Compute update statistics
        update_norms = []
        for weights in client_updates:
            norm = np.sqrt(sum(np.sum(w ** 2) for w in weights))
            update_norms.append(norm)
        
        data = {
            'round': round_num,
            'selected': selected_clients,
            'malicious_selected': [c for c in selected_clients if c in malicious_clients],
            'honest_selected': [c for c in selected_clients if c not in malicious_clients],
            'validation_losses': validation_losses,
            'update_norms': update_norms,
            'reputations': reputations.copy(),
            'kept_after_filtering': kept_after_filtering,
            'filtered_out': [c for c in selected_clients if c not in kept_after_filtering]
        }
        
        self.round_data.append(data)
        
    def print_round_summary(self, round_num, malicious_clients):
        """Print comprehensive summary for a round."""
        if round_num >= len(self.round_data):
            return
            
        data = self.round_data[round_num]
        
        print(f"\n{'='*80}")
        print(f"🔍 DETAILED DEBUG: Round {data['round'] + 1}")
        print(f"{'='*80}")
        
        # Selection summary
        print(f"\n📋 CLIENT SELECTION:")
        print(f"   Total selected: {len(data['selected'])}")
        print(f"   Selected clients: {sorted(data['selected'])}")
        print(f"   Malicious selected: {data['malicious_selected']} ({len(data['malicious_selected'])} clients)")
        print(f"   Honest selected: {data['honest_selected']} ({len(data['honest_selected'])} clients)")
        
        # Update statistics
        print(f"\n📊 UPDATE STATISTICS:")
        for i, (client_idx, norm) in enumerate(zip(data['selected'], data['update_norms'])):
            is_malicious = client_idx in malicious_clients
            marker = "⚠️  MALICIOUS" if is_malicious else "✅ HONEST   "
            print(f"   Client {client_idx:2d} {marker}: "
                  f"val_loss={data['validation_losses'][i]:.4f}, "
                  f"update_norm={norm:.2f}")
        
        # Filtering results
        print(f"\n🔧 KRUM FILTERING:")
        print(f"   Kept: {sorted(data['kept_after_filtering'])} ({len(data['kept_after_filtering'])} clients)")
        print(f"   Filtered out: {sorted(data['filtered_out'])} ({len(data['filtered_out'])} clients)")
        
        if data['filtered_out']:
            malicious_filtered = [c for c in data['filtered_out'] if c in malicious_clients]
            honest_filtered = [c for c in data['filtered_out'] if c not in malicious_clients]
            
            if data['malicious_selected']:
                detection_rate = len(malicious_filtered) / len(data['malicious_selected']) * 100
                print(f"   ✅ Malicious filtered: {malicious_filtered} ({len(malicious_filtered)}/{len(data['malicious_selected'])} = {detection_rate:.1f}%)")
            
            if honest_filtered:
                print(f"   ❌ Honest filtered (FALSE POSITIVES): {honest_filtered}")
        
        # Reputation summary
        print(f"\n📈 REPUTATION SUMMARY:")
        if data['malicious_selected']:
            mal_reps = [data['reputations'][c] for c in data['malicious_selected']]
            print(f"   Malicious avg reputation: {np.mean(mal_reps):.3f} (min={min(mal_reps):.3f}, max={max(mal_reps):.3f})")
        
        honest_all = [c for c in range(len(data['reputations'])) if c not in malicious_clients]
        if honest_all:
            honest_reps = [data['reputations'][c] for c in honest_all]
            print(f"   Honest avg reputation: {np.mean(honest_reps):.3f} (min={min(honest_reps):.3f}, max={max(honest_reps):.3f})")
        
        print(f"{'='*80}\n")
    
    def analyze_attack_effectiveness(self, malicious_clients):
        """Analyze how well the system detected attacks over time."""
        print(f"\n{'='*80}")
        print(f"📊 ATTACK DETECTION ANALYSIS")
        print(f"{'='*80}\n")
        
        detection_rates = []
        false_positive_rates = []
        
        for data in self.round_data:
            if not data['malicious_selected']:
                continue
                
            # How many malicious were filtered?
            malicious_filtered = len([c for c in data['filtered_out'] if c in malicious_clients])
            malicious_selected = len(data['malicious_selected'])
            detection_rate = malicious_filtered / malicious_selected if malicious_selected > 0 else 0
            
            # How many honest were filtered (false positives)?
            honest_filtered = len([c for c in data['filtered_out'] if c not in malicious_clients])
            honest_selected = len(data['honest_selected'])
            fp_rate = honest_filtered / honest_selected if honest_selected > 0 else 0
            
            detection_rates.append(detection_rate)
            false_positive_rates.append(fp_rate)
        
        if detection_rates:
            print(f"📈 Average malicious detection rate: {np.mean(detection_rates)*100:.1f}%")
            print(f"   Best round: {max(detection_rates)*100:.1f}%")
            print(f"   Worst round: {min(detection_rates)*100:.1f}%")
        
        if false_positive_rates:
            print(f"\n❌ Average false positive rate: {np.mean(false_positive_rates)*100:.1f}%")
        
        # Show reputation evolution
        print(f"\n📈 REPUTATION EVOLUTION:")
        if self.round_data:
            print(f"{'Round':<8} {'Mal Avg':<12} {'Honest Avg':<12} {'Gap':<8}")
            print(f"{'-'*40}")
            
            for data in self.round_data[::max(1, len(self.round_data)//10)]:  # Show ~10 samples
                if not malicious_clients:
                    continue
                    
                malicious_reps = [data['reputations'][c] for c in malicious_clients]
                honest_all = [c for c in range(len(data['reputations'])) if c not in malicious_clients]
                honest_reps = [data['reputations'][c] for c in honest_all]
                
                mal_avg = np.mean(malicious_reps)
                honest_avg = np.mean(honest_reps)
                gap = honest_avg - mal_avg
                
                print(f"{data['round']+1:<8d} {mal_avg:<12.3f} {honest_avg:<12.3f} {gap:+8.3f}")
        
        print(f"\n{'='*80}\n")
    
    def reset(self):
        """Reset tracker for new experiment."""
        self.round_data = []

# Global debug tracker
debug_tracker = DebugTracker()


def verify_byzantine_attack_applied(weights_before, weights_after, attack_type='signflip'):
    """Verify that Byzantine attack was actually applied to weights."""
    
    if attack_type == 'signflip':
        # Check if weights were flipped (should be negated)
        signs_match = []
        for w_before, w_after in zip(weights_before, weights_after):
            # Check if roughly w_after ≈ -w_before
            # Compute correlation
            flat_before = w_before.flatten()
            flat_after = w_after.flatten()
            
            # Perfect sign flip should give correlation ≈ -1
            correlation = np.corrcoef(flat_before, flat_after)[0, 1]
            signs_match.append(correlation < -0.8)  # Should be close to -1
        
        attack_applied = np.mean(signs_match) > 0.8  # 80% of layers should be flipped
        
        if config.DEBUG_MODE:
            print(f"      [Byzantine Attack Verification]")
            print(f"        Attack type: {attack_type}")
            print(f"        Layers with correct sign flip: {sum(signs_match)}/{len(signs_match)}")
            print(f"        Attack properly applied: {'✅ YES' if attack_applied else '❌ NO'}")
            
        return attack_applied
    
    return True  # For other attack types, assume correct


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def generate_synthetic_dataset(n_train=10000, n_test=2000, n_features=100, 
                               n_classes=10, difficulty='medium'):
    """
    Generate synthetic classification dataset for fast federated learning experiments.
    
    This is MUCH faster than CIFAR-10/MNIST while still testing all VFL-TS components:
    - Non-IID data splits work the same way
    - Byzantine attacks affect gradients identically
    - Reputation system behaves the same
    - BUT: 10-50× faster training!
    
    Args:
        n_train: Number of training samples
        n_test: Number of test samples
        n_features: Number of input features
        n_classes: Number of output classes
        difficulty: 'easy', 'medium', 'hard' - controls class separability
    
    Returns:
        (X_train, y_train), (X_test, y_test), num_classes, input_shape
    """
    print(f"\n📥 Generating synthetic classification dataset...")
    print(f"   Train samples: {n_train}, Test samples: {n_test}")
    print(f"   Features: {n_features}, Classes: {n_classes}")
    print(f"   Difficulty: {difficulty}")
    
    # Set difficulty parameters
    if difficulty == 'easy':
        n_informative = int(n_features * 0.8)  # 80% informative features
        n_redundant = int(n_features * 0.1)
        class_sep = 2.0  # Well-separated classes
    elif difficulty == 'medium':
        n_informative = int(n_features * 0.5)  # 50% informative
        n_redundant = int(n_features * 0.2)
        class_sep = 1.0  # Moderately separated
    else:  # hard
        n_informative = int(n_features * 0.3)  # 30% informative
        n_redundant = int(n_features * 0.3)
        class_sep = 0.5  # Harder to separate
    
    # Generate training data
    X_train, y_train = make_classification(
        n_samples=n_train,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        n_clusters_per_class=2,
        class_sep=class_sep,
        flip_y=0.01,  # 1% label noise (realistic)
        random_state=42
    )
    
    # Generate test data (different random state for independence)
    X_test, y_test = make_classification(
        n_samples=n_test,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        n_clusters_per_class=2,
        class_sep=class_sep,
        flip_y=0.01,
        random_state=43
    )
    
    # Convert to float32 (same as real datasets)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    # Normalize features (zero mean, unit variance)
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0) + 1e-8  # Avoid division by zero
    
    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std  # Use train statistics
    
    num_classes = n_classes
    input_shape = (n_features,)
    
    print(f"✅ Synthetic dataset generated")
    print(f"   Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"   Train class distribution: {np.bincount(y_train)}")
    print(f"   Feature mean: {X_train.mean():.3f}, std: {X_train.std():.3f}")
    
    return (X_train, y_train), (X_test, y_test), num_classes, input_shape



def load_dataset(dataset_name):
    """
    Load and preprocess datasets for federated learning.
    
    Args:
        dataset_name: 'MNIST', 'FASHION_MNIST', 'CIFAR10', or 'SYNTHETIC'
    
    Returns:
        (X_train, y_train), (X_test, y_test), num_classes, input_shape
    """
    print(f"\n📥 Loading {dataset_name} dataset...")
    
    if dataset_name == 'SYNTHETIC':
        # NEW: Use synthetic data for fast development
        return generate_synthetic_dataset(
            n_train=50000,
            n_test=2000,
            n_features=50,#100
            n_classes=10,
            difficulty='hard'
        )
    elif dataset_name == 'MNIST':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        num_classes = 10
        # Normalize and reshape
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
        input_shape = (28, 28, 1)
        
    elif dataset_name == 'FASHION_MNIST':
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        num_classes = 10
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
        input_shape = (28, 28, 1)
        
    elif dataset_name == 'CIFAR10':
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        num_classes = 10
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        input_shape = (32, 32, 3)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"✅ Dataset loaded: Train={len(X_train)}, Test={len(X_test)}, Classes={num_classes}")
    print(f"   Train class distribution: {np.bincount(y_train)}")
    
    return (X_train, y_train), (X_test, y_test), num_classes, input_shape


def create_non_iid_split(X, y, num_clients, alpha=0.5):
    """
    Create non-IID data split using Dirichlet distribution.
    
    Args:
        X: Training data
        y: Training labels
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter (lower = more non-IID)
    
    Returns:
        client_data: List of (X_client, y_client) tuples
    """
    print(f"\n🔀 Creating non-IID data split (α={alpha})...")
    
    num_classes = len(np.unique(y))
    client_data = []
    
    # Generate Dirichlet distribution for each class
    proportions = np.random.dirichlet(np.repeat(alpha, num_clients), num_classes)
    
    for client_idx in range(num_clients):
        client_indices = []
        
        for class_idx in range(num_classes):
            # Get all indices for this class
            class_indices = np.where(y == class_idx)[0]
            
            # Calculate how many samples this client gets from this class
            num_samples = int(proportions[class_idx, client_idx] * len(class_indices))
            
            # Randomly select samples
            selected_indices = np.random.choice(
                class_indices, 
                size=min(num_samples, len(class_indices)), 
                replace=False
            )
            client_indices.extend(selected_indices)
        
        # Shuffle and store client data
        client_indices = np.array(client_indices)
        np.random.shuffle(client_indices)
        
        client_data.append((X[client_indices], y[client_indices]))
        
        if config.VERBOSE and client_idx < 5:
            class_dist = np.bincount(y[client_indices], minlength=num_classes)
            print(f"   Client {client_idx}: {len(client_indices)} samples, "
                  f"distribution={class_dist}")
    
    print(f"✅ Non-IID split created for {num_clients} clients")
    return client_data


def create_iid_split(X, y, num_clients):
    """
    Create IID data split (uniform random distribution).
    
    Args:
        X: Training data
        y: Training labels  
        num_clients: Number of clients
    
    Returns:
        client_data: List of (X_client, y_client) tuples
    """
    print(f"\n🔀 Creating IID data split...")
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    # Split evenly among clients
    samples_per_client = len(X) // num_clients
    client_data = []
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else len(X)
        
        client_data.append((
            X_shuffled[start_idx:end_idx],
            y_shuffled[start_idx:end_idx]
        ))
    
    print(f"✅ IID split created: ~{samples_per_client} samples per client")
    return client_data


# ============================================================================
# SHARED VALIDATION SET CONSTRUCTION (Phase 0 of VFL-TS)
# ============================================================================

def construct_shared_validation_set(client_data, samples_per_client=10, max_size=500):
    """
    Construct shared validation set V from client contributions.
    Each client contributes a small subset of their data.
    
    Args:
        client_data: List of (X, y) tuples for each client
        samples_per_client: Number of samples each client contributes
        max_size: Maximum size of validation set
    
    Returns:
        (X_val, y_val): Shared validation set
    """
    print(f"\n🔧 Constructing shared validation set...")
    
    X_val_list = []
    y_val_list = []
    
    for client_idx, (X_client, y_client) in enumerate(client_data):
        if len(X_client) == 0:
            continue
        
        # Each client contributes min(samples_per_client, |D_i|) samples
        n_samples = min(samples_per_client, len(X_client))
        
        # Secure sampling (random without replacement)
        indices = np.random.choice(len(X_client), n_samples, replace=False)
        
        X_val_list.append(X_client[indices])
        y_val_list.append(y_client[indices])
    
    # Concatenate all contributions
    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)
    
    # Deduplicate (simple approach: keep unique indices)
    # For images, exact deduplication is expensive, so we just cap the size
    if len(X_val) > max_size:
        indices = np.random.choice(len(X_val), max_size, replace=False)
        X_val = X_val[indices]
        y_val = y_val[indices]
    
    print(f"✅ Shared validation set created: {len(X_val)} samples")
    print(f"   Class distribution: {np.bincount(y_val)}")
    
    return X_val, y_val


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

def create_model(input_shape, num_classes, model_type='AUTO'):
    """
    Create neural network model for federated learning.
    
    Args:
        input_shape: Input data shape
        num_classes: Number of output classes
        model_type: 'AUTO', 'CNN', or 'MLP'
    
    Returns:
        Compiled Keras model
    """

    # Auto-detect model type based on input shape
    if model_type == 'AUTO':
        if len(input_shape) == 1:
            model_type = 'MLP'  # 1D input → MLP
        else:
            model_type = 'CNN'  # 2D/3D input → CNN

    if model_type == 'MLP':
        # Simple Multi-Layer Perceptron (FAST for synthetic data)
        #print(f"   Creating MLP model for input shape {input_shape}")
        model = tf.keras.Sequential([ 
        tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.Dense(64, activation='relu'),  # 100→64: 6,400 params
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu'),  # 64→32: 2,048 params
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(num_classes, activation='softmax')  # 32→10: 320 params
        ])
        """ model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ]) """
    elif model_type == 'CNN':
        if input_shape[0] == 28:  # MNIST/Fashion-MNIST
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
        else:  # CIFAR-10 (ResNet-style)
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                
                tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ============================================================================
# MULTI-DIMENSIONAL SIGNAL COMPUTATION (Phase 1 of VFL-TS)
# ============================================================================

class ClientSignals:
    """
    Compute multi-dimensional signals for client selection (Equation 2 in paper).
    Signals: [L_local, |D|, ||Δw||, cos(Δw, Δw_global), σ²_grad, τ]
    """
    
    @staticmethod
    def compute_local_loss(model, X, y, sample_size=100):
        """Compute local validation loss (L_i^local)."""
        if len(X) == 0:
            return float('inf')
        
        # Use a sample for efficiency
        n = min(sample_size, len(X))
        indices = np.random.choice(len(X), n, replace=False)
        
        preds = model.predict(X[indices], verbose=0)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y[indices], preds
        ).numpy().mean()
        
        return float(loss)
    
    @staticmethod
    def compute_update_norm(weights_new, weights_old):
        """Compute ||Δw||_2."""
        delta_weights = [w_new - w_old for w_new, w_old in zip(weights_new, weights_old)]
        norm = np.sqrt(sum(np.sum(dw ** 2) for dw in delta_weights))
        return float(norm)
    
    @staticmethod
    def compute_cosine_similarity(delta_w1, delta_w2):
        """
        Compute cos(Δw_i, Δw_global) for gradient alignment.
        Higher value = client update aligns with global direction.
        """
        # Flatten both updates
        flat1 = np.concatenate([dw.flatten() for dw in delta_w1])
        flat2 = np.concatenate([dw.flatten() for dw in delta_w2])
        
        # Compute cosine similarity
        dot_product = np.dot(flat1, flat2)
        norm1 = np.linalg.norm(flat1)
        norm2 = np.linalg.norm(flat2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cos_sim = dot_product / (norm1 * norm2)
        return float(cos_sim)
    
    @staticmethod
    def compute_gradient_variance(model, X, y, sample_size=50):
        """
        Compute σ²_∇ℓ - variance of gradients across samples.
        High variance may indicate data heterogeneity or quality issues.
        """
        if len(X) == 0:
            return 0.0
        
        # Sample a subset for efficiency
        n = min(sample_size, len(X))
        indices = np.random.choice(len(X), n, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
        
        # Collect gradients for each sample
        gradients = []
        
        for i in range(len(X_sample)):
            with tf.GradientTape() as tape:
                pred = model(X_sample[i:i+1], training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(
                    y_sample[i:i+1], pred
                )
            
            grads = tape.gradient(loss, model.trainable_weights)
            
            # Flatten gradients
            flat_grad = np.concatenate([g.numpy().flatten() for g in grads if g is not None])
            gradients.append(flat_grad)
        
        # Compute variance
        if len(gradients) > 1:
            gradients = np.array(gradients)
            variance = np.var(gradients, axis=0).mean()
            return float(variance)
        
        return 0.0
    
    @staticmethod
    def compute_all_signals(model, X, y, weights_new, weights_old, 
                           global_delta, staleness):
        """
        Compute all 6 signals for a client (Equation 2).
        
        Returns:
            dict with keys: local_loss, data_size, update_norm, 
                           cos_similarity, grad_variance, staleness
        """
        # Signal 1: Local validation loss
        local_loss = ClientSignals.compute_local_loss(model, X, y)
        
        # Signal 2: Dataset size
        data_size = len(X)
        
        # Signal 3: Update magnitude
        delta_w = [w_new - w_old for w_new, w_old in zip(weights_new, weights_old)]
        update_norm = np.sqrt(sum(np.sum(dw ** 2) for dw in delta_w))
        
        # Signal 4: Cosine similarity with global update
        cos_sim = ClientSignals.compute_cosine_similarity(delta_w, global_delta)
        
        # Signal 5: Gradient variance (computationally expensive - use sample)
        # OPTIONAL: Set to 0.0 for speed: grad_var = 0.0
        grad_var = ClientSignals.compute_gradient_variance(model, X, y)
        
        # Signal 6: Staleness (rounds since last participation)
        # This is provided externally
        
        return {
            'local_loss': local_loss,
            'data_size': data_size,
            'update_norm': update_norm,
            'cos_similarity': cos_sim,
            'grad_variance': grad_var,
            'staleness': staleness
        }


# ============================================================================
# REPUTATION SYSTEM
# ============================================================================

class ReputationSystem:
    """
    FIXED: Less aggressive reputation system with death spiral prevention.
    """
    
    def __init__(self, num_clients, lambda_param=0.95, penalty=0.20):
        """
        Initialize reputation system.
        
        Args:
            num_clients: Total number of clients
            lambda_param: Temporal decay (0.95 = 5% decay per round)
            penalty: Penalty factor when filtered (0.20 = 20% reduction)
        """
        self.num_clients = num_clients
        self.lambda_param = lambda_param
        self.penalty = penalty
        
        # Initialize all clients with reputation = 1.0
        self.reputations = np.ones(num_clients)
        
        # Track participation
        self.last_participation = np.zeros(num_clients, dtype=int)
        self.participation_count = np.zeros(num_clients, dtype=int)
        
        # ✅ NEW: Track consecutive filter counts
        self.filter_count = {}
        self.kept_count = {}
    
    def update_reputation(self, client_idx, validation_loss, round_num, 
                         verification_passed=True):
        """
        FIXED: Update reputation with progressive penalties and generous rewards.
        
        Args:
            client_idx: Client index
            validation_loss: Loss on shared validation set
            round_num: Current round number
            verification_passed: True if kept by Krum/TrimmedMean
        """
        
        current_rep = self.reputations[client_idx]
        
        if verification_passed:
            # ============================================================
            # Client was KEPT → REWARD
            # ============================================================
            
            # Define thresholds based on dataset
            if config.DATASET in ['MNIST', 'FASHION_MNIST']:
                excellent_threshold = 0.5
                good_threshold = 1.0
                acceptable_threshold = 2.0
            elif config.DATASET == 'CIFAR10':
                excellent_threshold = 0.8
                good_threshold = 1.5
                acceptable_threshold = 2.5
            elif config.DATASET == 'SYNTHETIC':
                excellent_threshold = 0.3
                good_threshold = 0.6
                acceptable_threshold = 1.0
            else:
                excellent_threshold = 0.5
                good_threshold = 1.0
                acceptable_threshold = 2.0
            
            # ✅ GENEROUS rewards based on performance
            if validation_loss < excellent_threshold:
                reward = 0.08  # Excellent performance
            elif validation_loss < good_threshold:
                reward = 0.05  # Good performance
            elif validation_loss < acceptable_threshold:
                reward = 0.02  # Acceptable performance
            else:
                reward = 0.01  # Poor but kept
            
            new_reputation = min(1.0, current_rep + reward)
            
            # ✅ Track kept count and reset filter count
            self.kept_count[client_idx] = self.kept_count.get(client_idx, 0) + 1
            self.filter_count[client_idx] = 0  # Reset filter streak
            
        else:
            # ============================================================
            # Client was FILTERED OUT → PROGRESSIVE PENALTY
            # ============================================================
            
            # ✅ Progressive penalties - first time is gentle
            times_filtered = self.filter_count.get(client_idx, 0)
            
            if times_filtered == 0:
                # First time - might be false positive
                penalty_factor = 0.90  # Only 10% penalty
            elif times_filtered == 1:
                # Second time - more suspicious
                penalty_factor = 0.80  # 20% penalty
            elif times_filtered == 2:
                # Third time - likely malicious
                penalty_factor = 0.70  # 30% penalty
            else:
                # Repeatedly filtered - definitely malicious
                penalty_factor = 1.0 - self.penalty  # Full penalty (20%)
            
            new_reputation = current_rep * penalty_factor
            
            # Track consecutive filters
            self.filter_count[client_idx] = times_filtered + 1
            self.kept_count[client_idx] = 0  # Reset kept streak
        
        # ============================================================
        # Apply GENTLE temporal decay
        # ============================================================
        new_reputation *= self.lambda_param  # 0.95 = only 5% decay
        
        # ============================================================
        # Apply bounds with FLOOR at 0.01 (second chances)
        # ============================================================
        self.reputations[client_idx] = max(0.01, min(1.0, new_reputation))
        
        # Update participation tracking
        self.last_participation[client_idx] = round_num
        self.participation_count[client_idx] += 1
    
    def boost_consistent_performers(self, current_round):
        """
        ✅ NEW: Reward clients with consistent good behavior.
        Prevents death spiral from temporal decay.
        
        Call this every 3-5 rounds.
        """
        
        for client_idx in range(self.num_clients):
            # Check if client has been kept consistently
            kept_count = self.kept_count.get(client_idx, 0)
            filter_count = self.filter_count.get(client_idx, 0)
            
            # If kept 5+ times and rarely filtered, boost reputation
            if kept_count >= 5 and filter_count == 0:
                current_rep = self.reputations[client_idx]
                
                # Gradual boost toward 1.0
                if current_rep < 0.9:
                    boost = 0.03
                    self.reputations[client_idx] = min(1.0, current_rep + boost)
    
    def get_reputation(self, client_idx):
        """Get current reputation score for a client."""
        return self.reputations[client_idx]
    
    def get_staleness(self, client_idx, current_round):
        """Get staleness (rounds since last participation)."""
        return current_round - self.last_participation[client_idx]
    
    def get_all_reputations(self):
        """Get reputation scores for all clients."""
        return self.reputations.copy()



def check_for_divergence(round_num, current_accuracy, current_loss, 
                        history_accuracy, history_loss):
    """
    Detect model divergence.
    
    Returns:
        is_diverging: Boolean
        recovery_action: String describing action
    """
    
    if round_num < 3:
        return False, None
    
    prev_accuracy = history_accuracy[-1]
    prev_loss = history_loss[-1]
    
    # Check 1: Accuracy collapse
    if current_accuracy < 0.4 and prev_accuracy > 0.7:
        print(f"\n🚨 DIVERGENCE DETECTED: Accuracy collapse")
        print(f"   Round {round_num}: {prev_accuracy:.3f} → {current_accuracy:.3f}")
        return True, "accuracy_collapse"
    
    # Check 2: Loss explosion
    if current_loss > 10.0 and prev_loss < 1.0:
        print(f"\n🚨 DIVERGENCE DETECTED: Loss explosion")
        print(f"   Round {round_num}: {prev_loss:.3f} → {current_loss:.3f}")
        return True, "loss_explosion"
    
    # Check 3: NaN detection
    if np.isnan(current_loss) or np.isnan(current_accuracy):
        print(f"\n🚨 DIVERGENCE DETECTED: NaN at round {round_num}")
        return True, "nan_detected"
    
    return False, None


def handle_divergence_recovery(algorithm, round_num):
    """
    Handle divergence by resetting reputation system.
    
    Args:
        algorithm: VFLTSAlgorithm instance
        round_num: Current round
    """
    print(f"   🔄 Resetting reputation system at round {round_num}...")
    
    # Reset all low-reputation clients to moderate level
    for i in range(len(algorithm.reputation_system.reputations)):
        if algorithm.reputation_system.reputations[i] < 0.3:
            algorithm.reputation_system.reputations[i] = 0.5
    
    print(f"   ✅ Reputation system reset complete")
# ============================================================================
# CLIENT SELECTION (Phase 2 of VFL-TS)
# ============================================================================

def compute_quality_score(signals, global_weights=None, client_weights=None, selected_clients_data=None):
    """
    Compute quality score with PROPER normalization - ALL components in [0,1].
    
    Args:
        signals: Dictionary of client signals
        global_weights: Current global model weights (for gradient norm check)
        client_weights: Client's model weights (for gradient norm check)
        selected_clients_data: Dict of all selected clients' data for normalization
    
    Returns:
        Quality score (typically in range [0, 2.0])
    """
    
    # ============================================================
    # Component 1: Loss Score (normalized to [0, 1])
    # ============================================================
    if config.DATASET in ['MNIST', 'FASHION_MNIST']:
        max_reasonable_loss = 2.5
    elif config.DATASET == 'CIFAR10':
        max_reasonable_loss = 3.5
    elif config.DATASET == 'SYNTHETIC':
        max_reasonable_loss = 2.0
    else:
        max_reasonable_loss = 2.5
    
    # Lower loss = higher quality
    loss_score = max(0.0, 1.0 - (signals['local_loss'] / max_reasonable_loss))
    loss_score = min(1.0, loss_score)
    
    # ============================================================
    # Component 2: Data Size Score (normalized to [0, 1])
    # ============================================================
    # Use log normalization to prevent dominance
    data_score = np.log(1 + signals['data_size']) / np.log(1 + 10000)
    data_score = min(1.0, data_score)
    
    # ============================================================
    # Component 3: Gradient Alignment Score (normalized to [0, 1])
    # ============================================================
    # Map [-1, 1] → [0, 1]
    alignment_score = (signals['cos_similarity'] + 1.0) / 2.0
    alignment_score = max(0.0, min(1.0, alignment_score))
    
    # ============================================================
    # Component 4: Gradient Norm Sanity Check (penalty)
    # ============================================================
    norm_penalty = 0.0
    
    if global_weights is not None and client_weights is not None:
        # Compute this client's update norm
        local_norm = 0.0
        for cw, gw in zip(client_weights, global_weights):
            delta = cw - gw
            local_norm += np.sum(delta ** 2)
        local_norm = np.sqrt(local_norm)
        
        # Get reference norm from selected clients
        if selected_clients_data and len(selected_clients_data) > 1:
            all_norms = []
            for c_data in selected_clients_data.values():
                if 'update_norm' in c_data:
                    all_norms.append(c_data['update_norm'])
            
            if all_norms:
                median_norm = np.median(all_norms)
                
                if median_norm > 1e-8:
                    norm_ratio = local_norm / median_norm
                    
                    # Flag extreme outliers
                    if norm_ratio < 0.1 or norm_ratio > 10.0:
                        norm_penalty = 0.8
                    elif norm_ratio < 0.2 or norm_ratio > 5.0:
                        norm_penalty = 0.4
    
    # ============================================================
    # Combine Components
    # ============================================================
    quality = (config.QUALITY_WEIGHT_LOSS * loss_score + 
               config.QUALITY_WEIGHT_DATA * data_score + 
               config.QUALITY_WEIGHT_ALIGNMENT * alignment_score -
               norm_penalty)
    
    # Ensure in valid range
    max_quality = (config.QUALITY_WEIGHT_LOSS + 
                   config.QUALITY_WEIGHT_DATA + 
                   config.QUALITY_WEIGHT_ALIGNMENT)
    
    quality = max(0.0, min(max_quality, quality))
    
    # Debug logging
    if config.DEBUG_MODE and np.random.rand() < 0.05:
        print(f"      [Quality Score] Loss:{loss_score:.3f}, Data:{data_score:.3f}, "
              f"Align:{alignment_score:.3f}, NormPenalty:{norm_penalty:.3f}, Total:{quality:.3f}")
    
    return quality


def compute_selection_probability(signals, reputation, staleness):
    """
    Compute selection probability with dynamic reputation threshold.
    
    Args:
        signals: Dictionary of client signals
        reputation: Client reputation score (0-1)
        staleness: Rounds since last participation
    
    Returns:
        Selection probability (unnormalized)
    """
    
    # ✅ DYNAMIC threshold - stored as function attribute
    if not hasattr(compute_selection_probability, 'threshold'):
        compute_selection_probability.threshold = 0.05  # Start at 5%
    
    MIN_REPUTATION_THRESHOLD = compute_selection_probability.threshold
    
    if reputation < MIN_REPUTATION_THRESHOLD:
        return 0.0  # Hard cutoff
    
    # Quality component
    quality = compute_quality_score(signals)
    
    # Reputation with staleness penalty
    staleness_penalty = np.exp(-config.STALENESS_BETA * staleness)
    reputation_component = reputation * staleness_penalty
    
    # Combine
    probability = (
        config.SELECTION_ALPHA * quality +
        (1 - config.SELECTION_ALPHA) * reputation_component
    )
    
    return max(0.0, probability)

def select_clients_vflts(client_signals, reputation_system, current_round,
                         k, strategy='full'):
    """
    Two-stage client selection with death spiral prevention.
    
    Args:
        client_signals: List of signal dictionaries for each client
        reputation_system: ReputationSystem instance
        current_round: Current training round
        k: Number of clients to select
        strategy: 'full' (default), 'no_reputation', 'random'
    
    Returns:
        selected_indices: List of selected client indices
    """
    num_clients = len(client_signals)
    
    if strategy == 'random':
        available = [i for i in range(num_clients)
                    if client_signals[i] is not None]
        selected = np.random.choice(available, size=min(k, len(available)),
                                   replace=False)
        return selected.tolist()
    
    # ✅ STEP 1: Check for death spiral and reset if needed
    avg_reputation = np.mean(reputation_system.reputations)
    eligible_count = sum(1 for rep in reputation_system.reputations if rep >= 0.05)
    
    if eligible_count < num_clients * 0.5 or avg_reputation < 0.3:
        print(f"\n🚨 DEATH SPIRAL DETECTED at round {current_round}")
        print(f"   Eligible: {eligible_count}/{num_clients}, Avg reputation: {avg_reputation:.3f}")
        print(f"   🔄 RESETTING reputation system...")
        
        # Reset all to moderate level
        for i in range(num_clients):
            reputation_system.reputations[i] = 0.7
        
        # Clear filter counts
        if hasattr(reputation_system, 'filter_count'):
            reputation_system.filter_count = {}
        
        print(f"   ✅ All reputations reset to 0.7")
    
    # ✅ STEP 2: Adjust dynamic threshold based on system health
    if avg_reputation < 0.3:
        compute_selection_probability.threshold = 0.05  # Lower threshold
    elif avg_reputation < 0.5:
        compute_selection_probability.threshold = 0.08
    else:
        compute_selection_probability.threshold = 0.10  # Normal threshold
    
    # ✅ STEP 3: Compute selection probabilities
    probabilities = []
    
    if config.DEBUG_MODE and current_round % 5 == 0:
        print(f"\n      [Client Selection Debug - Round {current_round}]")
    
    for i in range(num_clients):
        if client_signals[i] is None:
            probabilities.append(0.0)
            continue
        
        # Get reputation and staleness
        reputation = reputation_system.get_reputation(i)
        staleness = reputation_system.get_staleness(i, current_round)
        
        if strategy == 'no_reputation':
            reputation = 1.0
            staleness = 0
        
        # Compute probability
        prob = compute_selection_probability(
            client_signals[i],
            reputation,
            staleness
        )
        
        probabilities.append(prob)
        
        if config.DEBUG_MODE and current_round % 5 == 0 and i < 3:
            quality = compute_quality_score(client_signals[i])
            print(f"        Client {i}: Quality={quality:.3f}, Rep={reputation:.3f}, "
                  f"Staleness={staleness}, Prob={prob:.4f}")
    
    # ✅ STEP 4: Normalize probabilities
    probabilities = np.array(probabilities)
    
    if config.DEBUG_MODE and current_round % 5 == 0:
        print(f"        Prob range: [{probabilities.min():.4f}, {probabilities.max():.4f}], "
              f"Sum: {probabilities.sum():.4f}")
    
    # Handle zero probabilities
    if probabilities.sum() < 1e-10:
        print(f"        ⚠️  WARNING: All probabilities zero, using uniform distribution")
        probabilities = np.ones(num_clients)
    
    # Normalize
    probabilities = probabilities / probabilities.sum()
    
    # ✅ STEP 5: Sample clients
    non_zero_count = np.sum(probabilities > 1e-10)
    actual_k = min(k, num_clients, non_zero_count)
    
    if actual_k < k and config.DEBUG_MODE:
        print(f"        ⚠️  WARNING: Only {non_zero_count} clients eligible, "
              f"selecting {actual_k} instead of {k}")
    
    selected = np.random.choice(
        num_clients,
        size=actual_k,
        replace=False,
        p=probabilities
    )
    
    if config.DEBUG_MODE and current_round % 5 == 0:
        print(f"        Selected clients: {sorted(selected.tolist())}")
    
    return selected.tolist()


# ============================================================================
# ATTACK SIMULATIONS
# ============================================================================

def apply_byzantine_attack(weights, global_weights, attack_type='signflip', amplification=3.0):
    """
    Apply Byzantine attacks to model weights.
    
    Args:
        weights: Local model weights after training
        global_weights: Current global model weights (REQUIRED for scaling)
        attack_type: 'signflip', 'gaussian', 'scaling'
        amplification: Amplification factor for signflip attack
    
    Returns:
        attacked_weights: Modified weights
    """
    attacked_weights = []
    
    for w, gw in zip(weights, global_weights):
        if attack_type == 'signflip':
            # Sign-flipping attack: multiply by -1
            attacked_w = -amplification * w
        
        elif attack_type == 'gaussian':
            # ✅ FIXED: Gaussian noise attack on GRADIENT
            # Compute the gradient (update)
            update = w - gw
            
            # Compute noise scale based on GRADIENT std, not weight std
            gradient_std = np.std(update)
            
            # Handle edge case where gradient is very small
            if gradient_std < 1e-8:
                noise_scale = 0.1  # Small fixed scale
            else:
                noise_scale = gradient_std * 50  # 50x gradient magnitude
            
            # Add Gaussian noise to the gradient
            noise = np.random.normal(0, noise_scale, w.shape)
            attacked_update = update + noise
            
            # Apply noisy gradient to global weights
            attacked_w = gw + attacked_update
            
            # Equivalent to:
            # attacked_w = w + noise
            # But noise is scaled by gradient, not weights
        
        elif attack_type == 'scaling':
            update = w - gw  # Compute the gradient update
            scale_factor = np.random.uniform(100, 200)  # Large scaling factor
            attacked_update = update * scale_factor  # Scale the update
            attacked_w = gw + attacked_update  # Apply scaled update
        
        else:
            attacked_w = w
        
        attacked_weights.append(attacked_w)
    
    return attacked_weights


def apply_label_flipping(X, y, flip_ratio=1.0):
    y_flipped = y.copy()
    num_classes = len(np.unique(y))
    
    if flip_ratio >= 0.99:
        # SYSTEMATIC COMPLETE FLIPPING (STRONGEST)
        # Shift all labels by 1: 0→1, 1→2, 2→3, ..., 9→0
        # This creates a model that CONSISTENTLY misclassifies
        y_flipped = (y + 1) % num_classes
        
    else:
        # PARTIAL SYSTEMATIC FLIPPING
        # Flip a subset, but do it systematically
        flip_indices = np.random.choice(
            len(y), 
            size=int(len(y) * flip_ratio), 
            replace=False
        )
        
        # Systematic shift (not random)
        for idx in flip_indices:
            y_flipped[idx] = (y[idx] + 1) % num_classes
    
    return X, y_flipped


def simulate_gradient_reversal(weights_current, weights_previous=None):
    """
    Gradient reversal attack (strategic adversary).
    
    Malicious clients estimate the global learning direction
    and send updates in the opposite direction.
    
    Args:
        weights_current: Current global model weights
        weights_previous: Previous round's weights (for direction estimation)
    
    Returns:
        reversed_weights: Weights opposing global learning direction
    """
    if weights_previous is None:
        # First round: add strong random noise
        noisy_weights = []
        for w in weights_current:
            noise = np.random.normal(0, np.std(w) * 0.5, w.shape)
            noisy_weights.append(w + noise)
        return noisy_weights
    
    # Estimate global update direction and reverse it
    reversed_weights = []
    for w_curr, w_prev in zip(weights_current, weights_previous):
        delta = w_curr - w_prev  # Global learning direction
        
        # Go in opposite direction with 3x amplification
        reversed_w = w_curr - 3.0 * delta
        reversed_weights.append(reversed_w)
    
    return reversed_weights


# ============================================================================
# ROBUST AGGREGATION (Phase 4-5 of VFL-TS)
# ============================================================================

def krum_filter_gradient_aware(client_weights, 
                               global_weights, 
                               validation_losses, 
                               n_keep, 
                               round_num=0):
    """
    Enhanced Krum filtering using EUCLIDEAN DISTANCE (robust to sign-flip) 
    with confidence-based adaptive filtering.
    
    Fixes critical vulnerability in cosine-based version against sign-flip attacks.
    Preserves all adaptive confidence innovations.
    """
    n_clients = len(client_weights)
    
    if n_clients <= n_keep:
        return list(range(n_clients))
    
    if config.DEBUG_MODE:
        print(f"\n      [Gradient-Aware Krum (Euclidean)] Filtering {n_clients} clients to keep {n_keep}")
        val_losses_str = [f"{vl:.3f}" for vl in validation_losses]
        print(f"        Validation losses: {val_losses_str}")
    
    # ============================================================================
    # STEP 1: Compute updates and pairwise EUCLIDEAN DISTANCES
    # ============================================================================
    
    def flatten_weights(weights):
        return np.concatenate([w.flatten() for w in weights])
    
    updates = []
    for client_w in client_weights:
        update = flatten_weights([cw - gw for cw, gw in zip(client_w, global_weights)])
        updates.append(update)
    
    # Compute pairwise Euclidean distances
    distances = np.zeros((n_clients, n_clients))
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            dist = np.linalg.norm(updates[i] - updates[j])
            distances[i, j] = dist
            distances[j, i] = dist
    
    # Average distance to other clients (excluding self)
    avg_distances = []
    for i in range(n_clients):
        mask = np.ones(n_clients, dtype=bool)
        mask[i] = False
        avg_dist = np.mean(distances[i, mask])
        avg_distances.append(avg_dist)
    
    avg_distances = np.array(avg_distances)
    
    # ============================================================================
    # STEP 2: Compute relative-median distance deviations
    # ============================================================================
    
    median_distance = np.median(avg_distances)
    
    if config.DEBUG_MODE:
        print(f"\n        [Distance Statistics]")
        print(f"        Median distance: {median_distance:.3f}")
        print(f"        Distance range: [{np.min(avg_distances):.3f}, {np.max(avg_distances):.3f}]")
    
    # ============================================================================
    # STEP 3: Compute suspicion scores
    # ============================================================================
    
    # Normalize validation losses
    mean_val_loss = np.mean(validation_losses)
    normalized_val_losses = validation_losses / (mean_val_loss + 1e-10)
    
    # Fixed higher weight on validation loss (critical for sign-flip resistance)
    val_loss_weight = 0.4  # Increased from 0.1 — malicious have higher loss
    
    suspicion_scores = []
    
    if config.DEBUG_MODE:
        print(f"\n        [Gradient Analysis]")
    
    for i in range(n_clients):
        # Clients FARTHER than median are suspicious
        relative_deviation = avg_distances[i] - median_distance
        
        suspicion = relative_deviation + val_loss_weight * normalized_val_losses[i]
        suspicion_scores.append(suspicion)
        
        if config.DEBUG_MODE:
            print(f"        Client {i}: avg_dist={avg_distances[i]:.3f}, "
                  f"deviation={relative_deviation:+.3f}, "
                  f"val_loss={validation_losses[i]:.3f}, "
                  f"suspicion={suspicion:.3f}")
    
    suspicion_scores = np.array(suspicion_scores)
    
    # ============================================================================
    # STEP 4: Adaptive filtering based on separation confidence
    # ============================================================================
    
    sorted_indices = np.argsort(suspicion_scores)  # Low suspicion first
    
    kept_indices = sorted_indices[:n_keep]
    filtered_indices = sorted_indices[n_keep:]
    
    kept_avg_dist = np.mean(avg_distances[kept_indices])
    filtered_avg_dist = np.mean(avg_distances[filtered_indices])
    separation = abs(kept_avg_dist - filtered_avg_dist)  # Larger separation = clearer clusters
    
    if config.DEBUG_MODE:
        print(f"\n        🎯 Kept (low suspicion): {kept_indices.tolist()}")
        print(f"        🚫 Filtered (high suspicion): {filtered_indices.tolist()}")
        print(f"\n        Kept avg distance: {kept_avg_dist:.3f}")
        print(f"        Filtered avg distance: {filtered_avg_dist:.3f}")
        print(f"        Separation: {separation:.3f}")
    
    # ============================================================================
    # STEP 5: Confidence-based adaptive filtering (your excellent innovation!)
    # ============================================================================
    
    LOW_CONFIDENCE_THRESHOLD = 0.05    # Adjusted slightly higher for distance scale
    MEDIUM_CONFIDENCE_THRESHOLD = 0.10
    HIGH_CONFIDENCE_THRESHOLD = 0.20
    
    if separation < LOW_CONFIDENCE_THRESHOLD:
        if config.DEBUG_MODE:
            print(f"⚠️  VERY LOW CONFIDENCE (sep={separation:.3f}) → Keeping ALL")
        return list(range(n_clients))
    
    elif separation < MEDIUM_CONFIDENCE_THRESHOLD:
        adaptive_keep = int(n_clients * 0.80) if round_num < 5 else int(n_clients * 0.75)
        if config.DEBUG_MODE:
            print(f"⚠️  LOW CONFIDENCE → Keeping {adaptive_keep}/{n_clients}")
        return sorted_indices[:adaptive_keep].tolist()
    
    elif separation < HIGH_CONFIDENCE_THRESHOLD:
        if config.DEBUG_MODE:
            print(f"        ✅ MEDIUM CONFIDENCE → Keeping {n_keep}")
        return kept_indices.tolist()
    
    else:
        aggressive_keep = max(int(n_clients * 0.60), n_keep - 2) if round_num >= 5 else n_keep
        if config.DEBUG_MODE:
            print(f"        🎯 HIGH CONFIDENCE → Keeping {aggressive_keep}")
        return sorted_indices[:aggressive_keep].tolist()



def clip_update_norms(client_weights, global_weights, max_norm_percentile=90):
    """
    Clip extreme update norms using robust median-based thresholds.
    More robust than percentile-based clipping against Byzantine attacks.
    """
    # Compute norms
    norms = []
    for client_w in client_weights:
        delta = [cw - gw for cw, gw in zip(client_w, global_weights)]
        norm = np.sqrt(sum(np.sum(d**2) for d in delta))
        norms.append(norm)
    
    norms = np.array(norms)
    
    # ROBUST CLIPPING: Use median + MAD instead of percentile
    median_norm = np.median(norms)
    mad = np.median(np.abs(norms - median_norm))  # Median Absolute Deviation
    
    # Detect attack type by variance
    norm_std = np.std(norms)
    coefficient_of_variation = norm_std / (median_norm + 1e-10)
    
    if coefficient_of_variation > 0.5:
        # High variance → likely Byzantine with amplification
        # Use tighter clipping (1.5× MAD)
        clip_threshold = median_norm + 1.5 * max(mad, median_norm * 0.1)
        attack_type = "amplified"
    else:
        # Low variance → likely coordinated or gradient reversal
        # Use moderate clipping (2.0× MAD)
        clip_threshold = median_norm + 2.0 * max(mad, median_norm * 0.1)
        attack_type = "coordinated"
    
    if config.DEBUG_MODE:
        print(f"\n      [Robust Norm Clipping]")
        print(f"        Norm range: [{min(norms):.2f}, {max(norms):.2f}]")
        print(f"        Median: {median_norm:.2f}, MAD: {mad:.2f}")
        print(f"        Variance coefficient: {coefficient_of_variation:.2f} ({attack_type})")
        print(f"        Clip threshold: {clip_threshold:.2f}")
    
    # Apply clipping
    clipped_weights = []
    clipped_count = 0
    
    for client_w, norm in zip(client_weights, norms):
        if norm > clip_threshold:
            scale = clip_threshold / norm
            delta = [cw - gw for cw, gw in zip(client_w, global_weights)]
            delta_clipped = [d * scale for d in delta]
            client_w_clipped = [gw + d for gw, d in zip(global_weights, delta_clipped)]
            clipped_weights.append(client_w_clipped)
            clipped_count += 1
            
            if config.DEBUG_MODE:
                print(f"        Clipped client: norm {norm:.2f} → {clip_threshold:.2f}")
        else:
            clipped_weights.append(client_w)
    
    if config.DEBUG_MODE and clipped_count > 0:
        print(f"        Total clipped: {clipped_count}/{len(client_weights)} clients")
    
    return clipped_weights, coefficient_of_variation  # Return variance metric too!

# Keep old function for potential comparison/ablation studies
def krum_filter_validation_only(updates, validation_losses, n_keep):
    """
    Original Krum filtering based only on validation losses.
    DEPRECATED: Use krum_filter_gradient_aware for Byzantine robustness.
    """
    n_clients = len(updates)
    if n_clients <= n_keep:
        return list(range(n_clients))
    
    # Original validation-loss-only logic
    distances = np.zeros((n_clients, n_clients))
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            dist = abs(validation_losses[i] - validation_losses[j])
            distances[i, j] = dist
            distances[j, i] = dist
    
    scores = []
    k_neighbors = min(config.KRUM_NEIGHBORS, n_clients - 1)
    for i in range(n_clients):
        neighbor_distances = np.sort(distances[i])[1:k_neighbors+1]
        score = neighbor_distances.sum()
        scores.append(score)
    
    filtered_indices = np.argsort(scores)[:n_keep].tolist()
    return filtered_indices


def aggregate_weights_robust(client_weights, 
                             client_indices, 
                             validation_losses,
                             use_filtering=True, 
                             global_weights=None, 
                             round_num=0):
    """
    Robust weighted aggregation with gradient-aware Krum filtering.

    Args:
        client_weights: List of weight lists from clients
        client_indices: Indices of participating clients
        validation_losses: Losses on shared validation set
        use_filtering: Whether to apply Krum filtering
        global_weights: Current global model weights (needed for gradient analysis)
        round_num: Current round number

    Returns:
        aggregated_weights: Aggregated model weights
        kept_indices: Indices of clients actually used (after filtering)
    """
    n_clients = len(client_weights)

    if n_clients == 0:
        return None, []

    if config.DEBUG_MODE:
        print(f"\n      [Robust Aggregation] Processing {n_clients} clients")
        print(f"        Client indices: {client_indices}")
        print(f"        Filtering enabled: {use_filtering}")

    # ✅ STEP 1: Clip extreme updates FIRST (CRITICAL!)
    norm_variance_signal = 0.0
    if use_filtering and global_weights is not None:
        client_weights, norm_variance_signal = clip_update_norms(
            client_weights, 
            global_weights, 
            max_norm_percentile=90
        )

    # ✅ STEP 2: Apply gradient-aware Krum filtering if enabled
    if use_filtering and n_clients > 3:
        # Keep top (1 - attack_ratio) of clients
        n_keep = max(1, int(n_clients * (1 - config.TRIM_RATIO)))
        
        if global_weights is not None:
            filter_indices = krum_filter_gradient_aware(
                client_weights,
                global_weights,
                validation_losses, 
                n_keep,
                round_num=round_num
            )
        else:
            # Fallback to validation-only if global_weights not provided
            print("        ⚠️  WARNING: global_weights not provided, using validation-only filtering")
            filter_indices = krum_filter_validation_only(
                client_weights,
                validation_losses,
                n_keep
            )

        # Filter clients
        filtered_out = [client_indices[i] for i in range(n_clients) if i not in filter_indices]
        client_weights = [client_weights[i] for i in filter_indices]
        kept_indices = [client_indices[i] for i in filter_indices]
        validation_losses = [validation_losses[i] for i in filter_indices]

        if config.DEBUG_MODE:
            print(f"        Kept {len(kept_indices)} clients: {kept_indices}")
            print(f"        Filtered out {len(filtered_out)} clients: {filtered_out}")
    else:
        kept_indices = client_indices
        if config.DEBUG_MODE:
            print(f"        No filtering applied (n_clients={n_clients})")

    # Weighted aggregation based on inverse validation loss
    weights_inv_loss = [1.0 / (1.0 + loss) for loss in validation_losses]
    total_weight = sum(weights_inv_loss)

    if config.DEBUG_MODE:
        print(f"        Aggregation weights: {[f'{w/total_weight:.3f}' for w in weights_inv_loss]}")

    # Aggregate each layer
    aggregated_weights = []
    for layer_idx in range(len(client_weights[0])):
        weighted_layer = np.zeros_like(client_weights[0][layer_idx])
        for client_w, weight in zip(client_weights, weights_inv_loss):
            weighted_layer += (weight / total_weight) * client_w[layer_idx]
        aggregated_weights.append(weighted_layer)

    return aggregated_weights, kept_indices


# ============================================================================
# FEDERATED LEARNING ALGORITHMS
# ============================================================================

class FederatedAlgorithm:
    """Base class for federated learning algorithms."""
    
    def __init__(self, name, input_shape, num_classes):
        self.name = name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.global_model = None
        self.history = defaultdict(list)
    
    def initialize_model(self):
        """Initialize global model."""
        self.global_model = create_model(self.input_shape, self.num_classes)
        return self.global_model
    
    def select_clients(self, round_num, client_data, k):
        """Select k clients for this round."""
        raise NotImplementedError
    
    def train_client(self, client_idx, client_data, global_weights):
        """Train local model for one client."""
        raise NotImplementedError
    
    def aggregate(self, local_updates, client_indices):
        """Aggregate local updates to global model."""
        raise NotImplementedError


class FedAvgAlgorithm(FederatedAlgorithm):
    """Standard FedAvg baseline."""

    def __init__(self, input_shape, num_classes):
        super().__init__('FedAvg', input_shape, num_classes)
        self.prev_global_weights = None  # ← ADD THIS LINE

    def select_clients(self, round_num, client_data, k):
        """Random client selection."""
        available = [i for i in range(len(client_data)) if len(client_data[i][0]) > 0]
        return np.random.choice(available, size=min(k, len(available)), replace=False).tolist()
    
    def train_client(self, client_idx, client_data, global_weights, 
                attack_type=None, is_malicious=False):
        X, y = client_data
        
        if len(X) == 0:
            return None, 0.0
        
        # ═══════════════════════════════════════════════════════════
        # ATTACK 2: Label Flipping (Data Poisoning)
        # ═══════════════════════════════════════════════════════════
        if attack_type == 'label_flipping' and is_malicious:
            X, y = apply_label_flipping(X, y)
        
        # Create and train local model
        local_model = create_model(self.input_shape, self.num_classes)
        local_model.set_weights(global_weights)
        
        # Train longer if doing label flipping (strengthen poison)
        epochs = config.EPOCHS_LOCAL * 2 if (is_malicious and attack_type == 'label_flipping') else config.EPOCHS_LOCAL
        
        start_time = time.time()
        local_model.fit(X, y, epochs=epochs, 
                    batch_size=config.BATCH_SIZE, verbose=0)
        training_time = time.time() - start_time
        
        weights = local_model.get_weights()
        
        # ═══════════════════════════════════════════════════════════
        # Apply Attacks After Training
        # ═══════════════════════════════════════════════════════════
        if is_malicious and attack_type:
            
            # ATTACK 2: Amplify label-flipping poison
            if attack_type == 'label_flipping':
                amplified_weights = []
                for w, w_global in zip(weights, global_weights):
                    delta = w - w_global
                    amplified_w = w_global + 5.0 * delta  # 5x amplification
                    amplified_weights.append(amplified_w)
                weights = amplified_weights
            
            # ATTACK 1: Byzantine sign-flip
            elif 'byzantine' in attack_type:
                attack_subtype = attack_type.split('_')[1]
                weights = apply_byzantine_attack(
                    weights, 
                    global_weights,
                    attack_subtype, 
                    amplification=3.0
                )
            
            # ATTACK 3: Gradient reversal
            elif attack_type == 'gradient_reversal':
                prev_weights = self.prev_global_weights if hasattr(self, 'prev_global_weights') else None
                weights = simulate_gradient_reversal(global_weights, prev_weights)
        
        return weights, training_time
    
    def aggregate(self, local_weights, client_indices, client_data):
        """Simple weighted averaging."""
        if not local_weights:
            return None
        
        # Weight by dataset size
        total_samples = sum(len(client_data[i][1]) for i in client_indices)
        
        aggregated = []
        for layer_idx in range(len(local_weights[0])):
            weighted_layer = np.zeros_like(local_weights[0][layer_idx])
            
            for i, client_idx in enumerate(client_indices):
                weight = len(client_data[client_idx][1]) / total_samples
                weighted_layer += weight * local_weights[i][layer_idx]
            
            aggregated.append(weighted_layer)
        
        return aggregated


class FedProxAlgorithm(FederatedAlgorithm):
    """FedProx: Federated Optimization with Proximal Term (Li et al., MLSys 2020)."""

    def __init__(self, input_shape, num_classes, mu=0.01):
        super().__init__('FedProx', input_shape, num_classes)
        self.mu = mu  # Proximal term coefficient
        self.prev_global_weights = None

    def select_clients(self, round_num, client_data, k):
        """Random client selection (same as FedAvg)."""
        available = [i for i in range(len(client_data)) if len(client_data[i][0]) > 0]
        return np.random.choice(available, size=min(k, len(available)), replace=False).tolist()

    def train_client(self, client_idx, client_data, global_weights,
                    attack_type=None, is_malicious=False):
        X, y = client_data

        if len(X) == 0:
            return None, 0.0

        # Apply label-flipping if this client is malicious
        if attack_type == 'label_flipping' and is_malicious:
            X, y = apply_label_flipping(X, y)

        # Create local model
        local_model = create_model(self.input_shape, self.num_classes)
        local_model.set_weights(global_weights)

        # Train longer if doing label flipping
        epochs = config.EPOCHS_LOCAL * 2 if (is_malicious and attack_type == 'label_flipping') else config.EPOCHS_LOCAL

        start_time = time.time()

        # Train with proximal regularization
        for _ in range(epochs):
            local_model.fit(X, y, epochs=1, batch_size=config.BATCH_SIZE, verbose=0)

            # Apply proximal term: pull weights back toward global model
            current_weights = local_model.get_weights()
            proximal_weights = []

            for w_local, w_global in zip(current_weights, global_weights):
                # Proximal update: w = w - (mu / 2) * (w - w_global)
                # Equivalent to: w = (1 - mu/2) * w + (mu/2) * w_global
                proximal_w = w_local - (self.mu / 2.0) * (w_local - w_global)
                proximal_weights.append(proximal_w)

            local_model.set_weights(proximal_weights)

        training_time = time.time() - start_time
        weights = local_model.get_weights()

        # Apply attacks after training (same as FedAvg)
        if is_malicious and attack_type:
            # ATTACK 2: Amplify label-flipping poison
            if attack_type == 'label_flipping':
                amplified_weights = []
                for w, w_global in zip(weights, global_weights):
                    delta = w - w_global
                    amplified_w = w_global + 5.0 * delta
                    amplified_weights.append(amplified_w)
                weights = amplified_weights

            # ATTACK 1: Byzantine attacks
            elif 'byzantine' in attack_type:
                attack_subtype = attack_type.split('_')[1]
                weights = apply_byzantine_attack(
                    weights,
                    global_weights,
                    attack_subtype,
                    amplification=3.0
                )

            # ATTACK 3: Gradient reversal
            elif attack_type == 'gradient_reversal':
                prev_weights = self.prev_global_weights if hasattr(self, 'prev_global_weights') else None
                weights = simulate_gradient_reversal(global_weights, prev_weights)

        return weights, training_time

    def aggregate(self, local_weights, client_indices, client_data):
        """Simple weighted averaging (same as FedAvg)."""
        if not local_weights:
            return None

        # Weight by dataset size
        total_samples = sum(len(client_data[i][1]) for i in client_indices)

        aggregated = []
        for layer_idx in range(len(local_weights[0])):
            weighted_layer = np.zeros_like(local_weights[0][layer_idx])

            for i, client_idx in enumerate(client_indices):
                weight = len(client_data[client_idx][1]) / total_samples
                weighted_layer += weight * local_weights[i][layer_idx]

            aggregated.append(weighted_layer)

        return aggregated


class TrimmedMeanAlgorithm(FederatedAlgorithm):
    """Trimmed Mean: Byzantine-robust aggregation via statistical trimming (Yin et al., ICML 2018)."""

    def __init__(self, input_shape, num_classes, trim_ratio=0.2):
        super().__init__('TrimmedMean', input_shape, num_classes)
        self.trim_ratio = trim_ratio  # Fraction to trim from each end (default 20%)
        self.prev_global_weights = None

    def select_clients(self, round_num, client_data, k):
        """Random client selection (same as FedAvg)."""
        available = [i for i in range(len(client_data)) if len(client_data[i][0]) > 0]
        return np.random.choice(available, size=min(k, len(available)), replace=False).tolist()

    def train_client(self, client_idx, client_data, global_weights,
                    attack_type=None, is_malicious=False):
        """Standard local training (same as FedAvg)."""
        X, y = client_data

        if len(X) == 0:
            return None, 0.0

        # Apply label-flipping if this client is malicious
        if attack_type == 'label_flipping' and is_malicious:
            X, y = apply_label_flipping(X, y)

        # Create and train local model
        local_model = create_model(self.input_shape, self.num_classes)
        local_model.set_weights(global_weights)

        # Train longer if doing label flipping
        epochs = config.EPOCHS_LOCAL * 2 if (is_malicious and attack_type == 'label_flipping') else config.EPOCHS_LOCAL

        start_time = time.time()
        local_model.fit(X, y, epochs=epochs,
                       batch_size=config.BATCH_SIZE, verbose=0)
        training_time = time.time() - start_time

        weights = local_model.get_weights()

        # Apply attacks after training (same as FedAvg)
        if is_malicious and attack_type:
            # ATTACK 2: Amplify label-flipping poison
            if attack_type == 'label_flipping':
                amplified_weights = []
                for w, w_global in zip(weights, global_weights):
                    delta = w - w_global
                    amplified_w = w_global + 5.0 * delta
                    amplified_weights.append(amplified_w)
                weights = amplified_weights

            # ATTACK 1: Byzantine attacks
            elif 'byzantine' in attack_type:
                attack_subtype = attack_type.split('_')[1]
                weights = apply_byzantine_attack(
                    weights,
                    global_weights,
                    attack_subtype,
                    amplification=3.0
                )

            # ATTACK 3: Gradient reversal
            elif attack_type == 'gradient_reversal':
                prev_weights = self.prev_global_weights if hasattr(self, 'prev_global_weights') else None
                weights = simulate_gradient_reversal(global_weights, prev_weights)

        return weights, training_time

    def aggregate(self, local_weights, _client_indices, _client_data):
        """Trimmed Mean aggregation: remove top/bottom values, then average."""
        if not local_weights:
            return None

        n_clients = len(local_weights)

        # Calculate number of values to trim from each end
        n_trim = max(1, int(n_clients * self.trim_ratio / 2))

        # Aggregate each layer using trimmed mean
        aggregated = []

        for layer_idx in range(len(local_weights[0])):
            # Get this layer from all clients
            layer_shape = local_weights[0][layer_idx].shape
            layer_values = np.array([w[layer_idx] for w in local_weights])

            # Flatten for coordinate-wise trimming
            n_params = np.prod(layer_shape)
            layer_flat = layer_values.reshape(n_clients, n_params)

            # Coordinate-wise trimmed mean
            trimmed_layer = np.zeros(n_params)

            for param_idx in range(n_params):
                # Get values for this parameter across all clients
                param_values = layer_flat[:, param_idx]

                # Sort values
                sorted_values = np.sort(param_values)

                # Trim top and bottom n_trim values
                if n_clients > 2 * n_trim:
                    trimmed_values = sorted_values[n_trim:-n_trim]
                else:
                    # If not enough clients, use all values
                    trimmed_values = sorted_values

                # Average the remaining values
                trimmed_layer[param_idx] = np.mean(trimmed_values)

            # Reshape back to original shape
            aggregated_layer = trimmed_layer.reshape(layer_shape)
            aggregated.append(aggregated_layer)

        return aggregated


class KrumAlgorithm(FederatedAlgorithm):
    """Krum: Byzantine-robust aggregation (Blanchard et al., NIPS 2017)."""

    def __init__(self, input_shape, num_classes, n_byzantine=None, multi_krum=True):
        super().__init__('Krum', input_shape, num_classes)
        self.n_byzantine = n_byzantine  # Maximum number of Byzantine clients
        self.multi_krum = multi_krum  # Use Multi-Krum (average top-m) vs single Krum
        self.prev_global_weights = None

    def select_clients(self, round_num, client_data, k):
        """Random client selection (same as FedAvg)."""
        available = [i for i in range(len(client_data)) if len(client_data[i][0]) > 0]
        return np.random.choice(available, size=min(k, len(available)), replace=False).tolist()

    def train_client(self, client_idx, client_data, global_weights,
                    attack_type=None, is_malicious=False):
        """Standard local training (same as FedAvg)."""
        X, y = client_data

        if len(X) == 0:
            return None, 0.0

        # Apply label-flipping if this client is malicious
        if attack_type == 'label_flipping' and is_malicious:
            X, y = apply_label_flipping(X, y)

        # Create and train local model
        local_model = create_model(self.input_shape, self.num_classes)
        local_model.set_weights(global_weights)

        # Train longer if doing label flipping
        epochs = config.EPOCHS_LOCAL * 2 if (is_malicious and attack_type == 'label_flipping') else config.EPOCHS_LOCAL

        start_time = time.time()
        local_model.fit(X, y, epochs=epochs,
                       batch_size=config.BATCH_SIZE, verbose=0)
        training_time = time.time() - start_time

        weights = local_model.get_weights()

        # Apply attacks after training (same as FedAvg)
        if is_malicious and attack_type:
            # ATTACK 2: Amplify label-flipping poison
            if attack_type == 'label_flipping':
                amplified_weights = []
                for w, w_global in zip(weights, global_weights):
                    delta = w - w_global
                    amplified_w = w_global + 5.0 * delta
                    amplified_weights.append(amplified_w)
                weights = amplified_weights

            # ATTACK 1: Byzantine attacks
            elif 'byzantine' in attack_type:
                attack_subtype = attack_type.split('_')[1]
                weights = apply_byzantine_attack(
                    weights,
                    global_weights,
                    attack_subtype,
                    amplification=3.0
                )

            # ATTACK 3: Gradient reversal
            elif attack_type == 'gradient_reversal':
                prev_weights = self.prev_global_weights if hasattr(self, 'prev_global_weights') else None
                weights = simulate_gradient_reversal(global_weights, prev_weights)

        return weights, training_time

    def aggregate(self, local_weights, client_indices, client_data):
        """Krum/Multi-Krum aggregation: select clients with smallest distance scores."""
        if not local_weights:
            return None

        n_clients = len(local_weights)

        # Set number of Byzantine clients (default: 20% of participants)
        if self.n_byzantine is None:
            f = max(1, n_clients // 5)
        else:
            f = self.n_byzantine

        # Need at least 2f+3 clients for Krum to work
        if n_clients < 2 * f + 3:
            # Fallback to simple averaging
            return self._simple_average(local_weights, client_indices, client_data)

        # Flatten weights for distance computation
        def flatten_weights(weights):
            return np.concatenate([w.flatten() for w in weights])

        flattened_weights = [flatten_weights(w) for w in local_weights]

        # Compute pairwise Euclidean distances
        distances = np.zeros((n_clients, n_clients))
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                dist = np.linalg.norm(flattened_weights[i] - flattened_weights[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # Compute Krum scores: sum of distances to n-f-2 closest neighbors
        n_close = n_clients - f - 2
        scores = []

        for i in range(n_clients):
            # Get distances to all other clients
            dists_to_others = distances[i].copy()
            dists_to_others[i] = np.inf  # Exclude self

            # Sum of distances to n_close nearest neighbors
            closest_dists = np.partition(dists_to_others, n_close - 1)[:n_close]
            score = np.sum(closest_dists)
            scores.append(score)

        scores = np.array(scores)

        if self.multi_krum:
            # Multi-Krum: select top m clients with smallest scores
            # m = n - f - 2 (keep majority of honest clients)
            m = max(1, n_clients - f - 2)
            selected_indices = np.argsort(scores)[:m]

            # Average the selected clients (weighted by dataset size)
            selected_weights = [local_weights[i] for i in selected_indices]
            selected_client_indices = [client_indices[i] for i in selected_indices]

            total_samples = sum(len(client_data[selected_client_indices[i]][1])
                              for i in range(len(selected_indices)))

            aggregated = []
            for layer_idx in range(len(selected_weights[0])):
                weighted_layer = np.zeros_like(selected_weights[0][layer_idx])

                for i, idx in enumerate(selected_indices):
                    client_idx = client_indices[idx]
                    weight = len(client_data[client_idx][1]) / total_samples
                    weighted_layer += weight * selected_weights[i][layer_idx]

                aggregated.append(weighted_layer)

            return aggregated
        else:
            # Single Krum: select the client with smallest score
            best_idx = np.argmin(scores)
            return local_weights[best_idx]

    def _simple_average(self, local_weights, client_indices, client_data):
        """Fallback to simple averaging when not enough clients."""
        total_samples = sum(len(client_data[i][1]) for i in client_indices)

        aggregated = []
        for layer_idx in range(len(local_weights[0])):
            weighted_layer = np.zeros_like(local_weights[0][layer_idx])

            for i, client_idx in enumerate(client_indices):
                weight = len(client_data[client_idx][1]) / total_samples
                weighted_layer += weight * local_weights[i][layer_idx]

            aggregated.append(weighted_layer)

        return aggregated


class FLTrustAlgorithm(FederatedAlgorithm):
    """FLTrust: Byzantine-robust FL with root dataset (Cao et al., NDSS 2021)."""

    def __init__(self, input_shape, num_classes, root_dataset):
        super().__init__('FLTrust', input_shape, num_classes)
        self.root_dataset = root_dataset  # Small trusted dataset (X_root, y_root)
        self.prev_global_weights = None

    def select_clients(self, round_num, client_data, k):
        """Random client selection (same as FedAvg)."""
        available = [i for i in range(len(client_data)) if len(client_data[i][0]) > 0]
        return np.random.choice(available, size=min(k, len(available)), replace=False).tolist()

    def train_client(self, client_idx, client_data, global_weights,
                    attack_type=None, is_malicious=False):
        """Standard local training (same as FedAvg)."""
        X, y = client_data

        if len(X) == 0:
            return None, 0.0

        # Apply label-flipping if this client is malicious
        if attack_type == 'label_flipping' and is_malicious:
            X, y = apply_label_flipping(X, y)

        # Create and train local model
        local_model = create_model(self.input_shape, self.num_classes)
        local_model.set_weights(global_weights)

        # Train longer if doing label flipping
        epochs = config.EPOCHS_LOCAL * 2 if (is_malicious and attack_type == 'label_flipping') else config.EPOCHS_LOCAL

        start_time = time.time()
        local_model.fit(X, y, epochs=epochs,
                       batch_size=config.BATCH_SIZE, verbose=0)
        training_time = time.time() - start_time

        weights = local_model.get_weights()

        # Apply attacks after training (same as FedAvg)
        if is_malicious and attack_type:
            # ATTACK 2: Amplify label-flipping poison
            if attack_type == 'label_flipping':
                amplified_weights = []
                for w, w_global in zip(weights, global_weights):
                    delta = w - w_global
                    amplified_w = w_global + 5.0 * delta
                    amplified_weights.append(amplified_w)
                weights = amplified_weights

            # ATTACK 1: Byzantine attacks
            elif 'byzantine' in attack_type:
                attack_subtype = attack_type.split('_')[1]
                weights = apply_byzantine_attack(
                    weights,
                    global_weights,
                    attack_subtype,
                    amplification=3.0
                )

            # ATTACK 3: Gradient reversal
            elif attack_type == 'gradient_reversal':
                prev_weights = self.prev_global_weights if hasattr(self, 'prev_global_weights') else None
                weights = simulate_gradient_reversal(global_weights, prev_weights)

        return weights, training_time

    def aggregate(self, local_weights, client_indices, client_data):
        """FLTrust aggregation: trust-based weighting using server's root dataset update."""
        if not local_weights:
            return None

        n_clients = len(local_weights)

        # Get current global weights (stored in self.global_model)
        global_weights = self.global_model.get_weights()

        # ========================================================================
        # STEP 1: Compute server update using root dataset
        # ========================================================================
        X_root, y_root = self.root_dataset

        # Train server model on root dataset
        server_model = create_model(self.input_shape, self.num_classes)
        server_model.set_weights(global_weights)
        server_model.fit(X_root, y_root, epochs=config.EPOCHS_LOCAL,
                        batch_size=config.BATCH_SIZE, verbose=0)
        server_weights = server_model.get_weights()

        # Compute server update (gradient)
        server_update = self._flatten_weights([sw - gw for sw, gw in zip(server_weights, global_weights)])

        # Normalize server update
        server_norm = np.linalg.norm(server_update)
        if server_norm < 1e-10:
            # Server update is zero, fallback to simple averaging
            return self._simple_average(local_weights, client_indices, client_data)

        # ========================================================================
        # STEP 2: Compute trust scores for each client
        # ========================================================================
        trust_scores = []

        for client_w in local_weights:
            # Compute client update
            client_update = self._flatten_weights([cw - gw for cw, gw in zip(client_w, global_weights)])

            # Compute cosine similarity with server update
            client_norm = np.linalg.norm(client_update)

            if client_norm < 1e-10:
                # Client update is zero
                cos_sim = 0.0
            else:
                cos_sim = np.dot(server_update, client_update) / (server_norm * client_norm)

            # Apply ReLU: trust score = max(0, cosine_similarity)
            trust_score = max(0.0, cos_sim)
            trust_scores.append(trust_score)

        trust_scores = np.array(trust_scores)

        # ========================================================================
        # STEP 3: Normalize trust scores
        # ========================================================================
        total_trust = np.sum(trust_scores)

        if total_trust < 1e-10:
            # All trust scores are zero or negative, fallback to simple averaging
            return self._simple_average(local_weights, client_indices, client_data)

        # Normalize trust scores to sum to 1
        normalized_trust_scores = trust_scores / total_trust

        # ========================================================================
        # STEP 4: Compute trust-weighted server update magnitude
        # ========================================================================
        # Clip client updates by server update magnitude (as per FLTrust paper)
        clipped_updates = []

        for client_w in local_weights:
            client_update = [cw - gw for cw, gw in zip(client_w, global_weights)]
            client_update_flat = self._flatten_weights(client_update)
            client_norm = np.linalg.norm(client_update_flat)

            # Clip client update magnitude to server update magnitude
            if client_norm > server_norm:
                clip_factor = server_norm / client_norm
                clipped_update = [cu * clip_factor for cu in client_update]
            else:
                clipped_update = client_update

            clipped_updates.append(clipped_update)

        # ========================================================================
        # STEP 5: Trust-weighted aggregation
        # ========================================================================
        aggregated = []

        for layer_idx in range(len(global_weights)):
            # Start with zero
            weighted_layer = np.zeros_like(global_weights[layer_idx])

            # Add trust-weighted updates
            for i in range(n_clients):
                weighted_layer += normalized_trust_scores[i] * clipped_updates[i][layer_idx]

            # Apply to global weights
            aggregated.append(global_weights[layer_idx] + weighted_layer)

        return aggregated

    def _flatten_weights(self, weights):
        """Flatten weight list to 1D array."""
        return np.concatenate([w.flatten() for w in weights])

    def _simple_average(self, local_weights, client_indices, client_data):
        """Fallback to simple averaging."""
        total_samples = sum(len(client_data[i][1]) for i in client_indices)

        aggregated = []
        for layer_idx in range(len(local_weights[0])):
            weighted_layer = np.zeros_like(local_weights[0][layer_idx])

            for i, client_idx in enumerate(client_indices):
                weight = len(client_data[client_idx][1]) / total_samples
                weighted_layer += weight * local_weights[i][layer_idx]

            aggregated.append(weighted_layer)

        return aggregated


class VFLTSAlgorithm(FederatedAlgorithm):
    """VFL-TS: Verifiable Federated Learning with Two-Stage Selection."""
    
    def __init__(self, input_shape, num_classes, shared_val_set, 
                 ablation=None):
        super().__init__('VFL-TS', input_shape, num_classes)
        self.shared_val_set = shared_val_set
        self.reputation_system = None
        self.ablation = ablation  # For ablation studies
        self.malicious_clients = set()  # Track malicious clients for debugging
        
        # Track average global update for cosine similarity
        self.prev_global_weights = None
        self.global_delta = None
    
    def initialize(self, num_clients):
        """Initialize reputation system."""
        self.reputation_system = ReputationSystem(
            num_clients,
            lambda_param=config.REPUTATION_LAMBDA,
            penalty=config.REPUTATION_PENALTY
        )
        self.global_model = create_model(self.input_shape, self.num_classes)
        self.prev_global_weights = self.global_model.get_weights()
        return self.global_model
    
    def compute_client_signals(self, client_idx, client_data, global_weights,
                               current_round):
        """
        Compute all signals for a client (Phase 1).
        Uses lightweight signal computation (1 epoch training).
        """
        X, y = client_data

        if len(X) == 0:
            return None

        # Create temporary model for signal computation
        temp_model = create_model(self.input_shape, self.num_classes)
        temp_model.set_weights(global_weights)

        # Train for only 1 epoch for signal estimation
        temp_model.fit(X, y, epochs=1,
                      batch_size=config.BATCH_SIZE, verbose=0)

        new_weights = temp_model.get_weights()

        # Get staleness
        staleness = self.reputation_system.get_staleness(client_idx, current_round)

        # Compute global delta (average from previous round)
        if self.global_delta is None:
            self.global_delta = [np.zeros_like(w) for w in global_weights]

        # Compute all signals
        signals = ClientSignals.compute_all_signals(
            temp_model, X, y, new_weights, global_weights,
            self.global_delta, staleness
        )

        return signals
    
    def select_clients(self, round_num, client_data, k):
        """Two-stage selection (Phase 2)."""
        # Compute signals for all clients
        global_weights = self.global_model.get_weights()
        
        client_signals = []
        for i in range(len(client_data)):
            signals = self.compute_client_signals(
                i, client_data[i], global_weights, round_num
            )
            client_signals.append(signals)
        
        # Stage 1: Signal-based probabilistic selection
        strategy = 'full'
        if self.ablation == 'random_selection':
            strategy = 'random'
        elif self.ablation == 'no_reputation':
            strategy = 'no_reputation'
        
        selected = select_clients_vflts(
            client_signals, 
            self.reputation_system,
            round_num, 
            k,
            strategy=strategy
        )
        
        return selected
    
    def train_client(self, client_idx, client_data, global_weights,
                    attack_type=None, is_malicious=False):
        """Train local model and evaluate on shared validation set."""
        X, y = client_data
        
        if len(X) == 0:
            return None, 0.0, float('inf')
        
        # Apply label-flipping if this client is malicious
        if attack_type == 'label_flipping' and is_malicious:
            X, y = apply_label_flipping(X, y)
        
        # Train local model
        local_model = create_model(self.input_shape, self.num_classes)
        local_model.set_weights(global_weights)
        
        start_time = time.time()
        local_model.fit(X, y, epochs=config.EPOCHS_LOCAL,
                       batch_size=config.BATCH_SIZE, verbose=0)
        training_time = time.time() - start_time
        
        weights = local_model.get_weights()
        weights_before_attack = [w.copy() for w in weights]  # For verification
        
        # Evaluate on shared validation set
        X_val, y_val = self.shared_val_set
        val_preds = local_model.predict(X_val, verbose=0)
        val_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_val, val_preds
        ).numpy().mean()
        
        # Apply Byzantine attacks after training
        if is_malicious and attack_type:

            # ATTACK 2: Amplify label-flipping poison
            if attack_type == 'label_flipping':
                amplified_weights = []
                for w, w_global in zip(weights, global_weights):
                    delta = w - w_global
                    amplified_w = w_global + 5.0 * delta  # 5x amplification
                    amplified_weights.append(amplified_w)
                weights = amplified_weights
            elif 'byzantine' in attack_type:
                attack_subtype = attack_type.split('_')[1]
                weights = apply_byzantine_attack(
                    weights, 
                    global_weights,
                    attack_subtype,
                    amplification=3.0
                )
                
                # ✅ Verify attack was applied (sample 20% of malicious clients)
                if config.DEBUG_MODE and np.random.rand() < 0.2:
                    verify_byzantine_attack_applied(weights_before_attack, weights, attack_subtype)
                    
            elif attack_type == 'gradient_reversal':
                prev_weights = self.prev_global_weights if hasattr(self, 'prev_global_weights') else None
                weights = simulate_gradient_reversal(global_weights, prev_weights)

        
        return weights, training_time, val_loss
    
    def aggregate(self, local_weights, client_indices, validation_losses,
              client_data, round_num=0):
        """
        Robust aggregation with gradient-aware Krum filtering (Phases 4-5).
        """
        if not local_weights:
            return None

        # Stage 2: Robust filtering
        use_filtering = (self.ablation != 'no_robust_filtering')

        # ✅ NEW: Get current global weights for gradient analysis
        global_weights_ref = self.global_model.get_weights()

        # ✅ DEBUG: Log data before filtering
        if config.DEBUG_MODE and round_num % 5 == 0:
            print(f"\n      [Pre-Filtering Data - Round {round_num}]")
            print(f"        Clients: {client_indices}")
            print(f"        Val losses: {[f'{v:.4f}' for v in validation_losses]}")
            
            # Compute update norms for inspection
            update_norms = []
            for weights in local_weights:
                norm = np.sqrt(sum(np.sum(w ** 2) for w in weights))
                update_norms.append(norm)
            print(f"        Update norms: {[f'{n:.2f}' for n in update_norms]}")

        # ✅ MODIFIED: Pass global_weights to aggregation
        aggregated, kept_indices = aggregate_weights_robust(
            local_weights,
            client_indices,
            validation_losses,
            use_filtering=use_filtering,
            global_weights=global_weights_ref,
            round_num=round_num  # ✅ NEW PARAMETER
        )

        # Update reputations (Phase 6)
        filtered_out = [client_indices[i] for i in range(len(client_indices))
                    if client_indices[i] not in kept_indices]

        if config.DEBUG_MODE and round_num % 5 == 0:
            print(f"\n      [Reputation Update - Round {round_num}]")
            print(f"        Clients kept for aggregation: {kept_indices}")
            if filtered_out:
                print(f"        Clients filtered out (reputation penalty): {filtered_out}")

        for i, client_idx in enumerate(client_indices):
            kept = client_idx in kept_indices
            self.reputation_system.update_reputation(
                client_idx,
                validation_losses[i],
                round_num=round_num,
                verification_passed=kept
            )

            if config.DEBUG_MODE and round_num % 5 == 0 and i < 5:
                new_rep = self.reputation_system.get_reputation(client_idx)
                is_mal = "⚠️  MAL" if client_idx in self.malicious_clients else "✅ HON"
                print(f"        Client {client_idx} {is_mal}: val_loss={validation_losses[i]:.3f}, "
                    f"kept={kept}, new_rep={new_rep:.3f}")

        # ✅ DEBUG: Log to tracker
        if config.DEBUG_MODE:
            debug_tracker.log_round(
                round_num,
                client_indices,
                self.malicious_clients,
                local_weights,
                validation_losses,
                self.reputation_system.get_all_reputations(),
                kept_indices
            )

        # Track global delta for next round
        if aggregated and self.prev_global_weights:
            self.global_delta = [
                agg - prev for agg, prev in zip(aggregated, self.prev_global_weights)
            ]
            self.prev_global_weights = [w.copy() for w in aggregated]

        return aggregated


# ============================================================================
# TRAINING LOOP
# ============================================================================

def run_federated_training(algorithm, client_data, X_test, y_test,
                           attack_type=None, attack_ratio=0.0, run_id=0):
    """
    Run complete federated learning training with comprehensive debugging.
    FIXED VERSION with death spiral prevention and divergence detection.
    
    Args:
        algorithm: FederatedAlgorithm instance
        client_data: List of (X, y) tuples for each client
        X_test: Test data
        y_test: Test labels
        attack_type: Type of attack (None, 'byzantine_*', 'label_flipping', 'gradient_reversal')
        attack_ratio: Fraction of malicious clients
        run_id: Run identifier for logging
    
    Returns:
        history: Training metrics over rounds
    """
    print(f"\n{'='*80}")
    print(f"🚀 Running: {algorithm.name}")
    print(f"   Attack: {attack_type or 'None'}, Ratio: {attack_ratio}")
    print(f"   Run: {run_id + 1}/{config.NUM_RUNS}")
    print(f"{'='*80}\n")
    
    # Reset debug tracker
    debug_tracker.reset()
    
    # Initialize
    if hasattr(algorithm, 'initialize'):
        algorithm.initialize(len(client_data))
    else:
        algorithm.initialize_model()
    
    # Determine which clients are malicious
    num_malicious = int(len(client_data) * attack_ratio)
    malicious_clients = set(np.random.choice(
        len(client_data), num_malicious, replace=False
    )) if attack_ratio > 0 else set()
    
    # Pass malicious clients to algorithm for debugging
    if isinstance(algorithm, VFLTSAlgorithm):
        algorithm.malicious_clients = malicious_clients
    
    # Initialize history
    history = {
        'accuracy': [],
        'loss': [],
        'training_time': [],
        'selected_clients': [],
        'malicious_selected': [],
        'kept_after_filtering': [],
        'reputations': [],
        'malicious_clients': list(malicious_clients),
    }
    
    if malicious_clients:
        print(f"⚠️  Malicious clients: {sorted(malicious_clients)}")
    
    # Training loop
    for round_num in range(config.NUM_ROUNDS):
        round_start = time.time()
        
        # Select clients
        k = max(1, int(len(client_data) * config.CLIENT_PARTICIPATION_RATE))
        selected_indices = algorithm.select_clients(round_num, client_data, k)
        
        # Count malicious clients selected
        malicious_selected = len([i for i in selected_indices if i in malicious_clients])
        malicious_in_selection = [i for i in selected_indices if i in malicious_clients]

        if config.VERBOSE or config.DEBUG_MODE:
            if round_num % 5 == 0 or config.DEBUG_MODE:
                print(f"\n📍 Round {round_num + 1}/{config.NUM_ROUNDS}")
                print(f"   Selected {len(selected_indices)} clients: {sorted(selected_indices)}")
                if malicious_selected > 0:
                    print(f"   ⚠️  {malicious_selected} malicious clients selected: {malicious_in_selection}")
        
        # Train selected clients
        local_weights = []
        validation_losses = []
        
        for client_idx in selected_indices:
            # Determine if this specific client is malicious
            is_malicious = client_idx in malicious_clients
            
            # Train client
            if isinstance(algorithm, VFLTSAlgorithm):
                weights, _, val_loss = algorithm.train_client(
                    client_idx,
                    client_data[client_idx],
                    algorithm.global_model.get_weights(),
                    attack_type,
                    is_malicious
                )
                validation_losses.append(val_loss)
            else:
                weights, _ = algorithm.train_client(
                    client_idx,
                    client_data[client_idx],
                    algorithm.global_model.get_weights(),
                    attack_type,
                    is_malicious
                )
            
            if weights is not None:
                local_weights.append(weights)
        
        # ✅ Store previous global weights BEFORE aggregation
        if hasattr(algorithm, 'prev_global_weights'):
            algorithm.prev_global_weights = [w.copy() for w in algorithm.global_model.get_weights()]
        
        # Aggregate
        if local_weights:
            if isinstance(algorithm, VFLTSAlgorithm):
                aggregated = algorithm.aggregate(
                    local_weights, selected_indices,
                    validation_losses, client_data, round_num
                )
            else:
                aggregated = algorithm.aggregate(
                    local_weights, selected_indices, client_data
                )

            if aggregated:
                algorithm.global_model.set_weights(aggregated)
        
        # ✅ NEW: Boost consistent performers every 3 rounds
        if round_num > 0 and round_num % 3 == 0 and isinstance(algorithm, VFLTSAlgorithm):
            if config.DEBUG_MODE and round_num % 5 == 0:
                print(f"\n      [Reputation Boost] Rewarding consistent performers at round {round_num + 1}")
            algorithm.reputation_system.boost_consistent_performers(round_num)
        
        # Evaluate
        test_loss, test_acc = algorithm.global_model.evaluate(
            X_test, y_test, verbose=0
        )
        
        # ✅ NEW: Check for divergence
        if isinstance(algorithm, VFLTSAlgorithm):
            is_diverging, recovery_action = check_for_divergence(
                round_num,
                test_acc,
                test_loss,
                history['accuracy'],
                history['loss']
            )
            
            if is_diverging:
                print(f"   🔄 Triggering reputation system reset...")
                # Reset all clients to moderate reputation
                for i in range(len(algorithm.reputation_system.reputations)):
                    algorithm.reputation_system.reputations[i] = 0.7
                
                # Clear filter counts
                if hasattr(algorithm.reputation_system, 'filter_count'):
                    algorithm.reputation_system.filter_count = {}
                if hasattr(algorithm.reputation_system, 'kept_count'):
                    algorithm.reputation_system.kept_count = {}
                
                print(f"   ✅ Reputation system reset complete")
                
                # Optionally stop training for severe divergence
                if recovery_action in ["nan_detected"]:
                    print(f"   🛑 Stopping training due to severe divergence")
                    break
        
        # Record metrics
        history['accuracy'].append(test_acc)
        history['loss'].append(test_loss)
        history['training_time'].append(time.time() - round_start)
        history['selected_clients'].append(list(selected_indices))
        history['malicious_selected'].append(list(malicious_in_selection))
        
        # Track reputations
        if isinstance(algorithm, VFLTSAlgorithm):
            current_reps = algorithm.reputation_system.get_all_reputations()
            history['reputations'].append(current_reps.copy())

            # Show reputation summary periodically
            if config.VERBOSE and (round_num + 1) % 5 == 0:
                print(f"\n   📊 Round {round_num + 1} Summary:")
                print(f"      Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")
                if malicious_clients:
                    malicious_reps = [current_reps[i] for i in malicious_clients]
                    honest_reps = [current_reps[i] for i in range(len(client_data))
                                  if i not in malicious_clients]
                    print(f"      Avg Malicious Reputation: {np.mean(malicious_reps):.3f}")
                    print(f"      Avg Honest Reputation: {np.mean(honest_reps):.3f}")
        else:
            history['reputations'].append(np.zeros(len(client_data)))

        if config.VERBOSE and (round_num + 1) % 10 == 0:
            if not isinstance(algorithm, VFLTSAlgorithm):
                print(f"   Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")
    
    # ✅ DEBUG: Print detailed analysis after training
    if config.DEBUG_MODE and isinstance(algorithm, VFLTSAlgorithm) and attack_type:
        print(f"\n{'#'*80}")
        print(f"# DETAILED ROUND-BY-ROUND ANALYSIS")
        print(f"# Attack: {attack_type}, Ratio: {attack_ratio}")
        print(f"{'#'*80}")
        
        # Show first 3 rounds
        print(f"\n=== EARLY ROUNDS (Learning Phase) ===")
        for r in [0, 1, 2]:
            if r < config.NUM_ROUNDS:
                debug_tracker.print_round_summary(r, malicious_clients)
        
        # Show middle and late rounds
        print(f"\n=== MID/LATE ROUNDS (Detection Phase) ===")
        for r in [config.NUM_ROUNDS//3, 2*config.NUM_ROUNDS//3, config.NUM_ROUNDS-1]:
            if r < config.NUM_ROUNDS:
                debug_tracker.print_round_summary(r, malicious_clients)
        
        # Overall analysis
        if malicious_clients:
            debug_tracker.analyze_attack_effectiveness(malicious_clients)
    
    print(f"\n✅ Training complete: Final accuracy = {history['accuracy'][-1]:.4f}")
    
    return history


# ============================================================================
# EXPERIMENT EXECUTION
# ============================================================================

# ============================================================================
# EXPERIMENT EXECUTION
# ============================================================================

def run_all_experiments():
    """Run complete experimental suite."""
    print("\n" + "="*80)
    print("🔬 VFL-TS EXPERIMENTAL FRAMEWORK - COMPREHENSIVE DEBUG VERSION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Dataset: {config.DATASET}")
    print(f"  Clients: {config.NUM_CLIENTS}")
    print(f"  Rounds: {config.NUM_ROUNDS}")
    print(f"  Non-IID α: {config.NON_IID_ALPHA}")
    print(f"  Runs per config: {config.NUM_RUNS}")
    print(f"  Reputation λ: {config.REPUTATION_LAMBDA}")
    print(f"  Reputation penalty: {config.REPUTATION_PENALTY}")
    print(f"  Krum neighbors: {config.KRUM_NEIGHBORS}")
    print(f"  Trim ratio: {config.TRIM_RATIO}")
    print(f"  Debug mode: {config.DEBUG_MODE}")
    print(f"  Output directory: {config.OUTPUT_DIR}")
    print("\n" + "="*80)
    
    # Load dataset
    (X_train, y_train), (X_test, y_test), num_classes, input_shape = load_dataset(
        config.DATASET
    )
    
    # Create client data split
    if config.USE_NON_IID:
        client_data = create_non_iid_split(
            X_train, y_train, 
            config.NUM_CLIENTS, 
            config.NON_IID_ALPHA
        )
    else:
        client_data = create_iid_split(X_train, y_train, config.NUM_CLIENTS)
    
    # Construct shared validation set
    shared_val_set = construct_shared_validation_set(
        client_data,
        config.SHARED_VAL_SAMPLES_PER_CLIENT,
        config.SHARED_VAL_MAX_SIZE
    )

    # Create root dataset for FLTrust (use 30% of shared validation set)
    X_shared, y_shared = shared_val_set
    root_size = max(100, int(len(X_shared) * 0.3))  # At least 100 samples, or 30% of shared set
    root_dataset = (X_shared[:root_size], y_shared[:root_size])

    # Storage for all results
    all_results = {}

    # Accumulator for per-run folder output: {(algorithm, run_id): {attack_key: {ratio_key: history}}}
    per_run_results = {}

    def to_json_serializable(obj):
        """Recursively convert numpy types to native Python for JSON."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [to_json_serializable(item) for item in obj]
        return obj
    
    # Run experiments
    total_experiments = (
        len(config.ALGORITHMS) * 
        len(config.ATTACK_TYPES) * 
        len(config.ATTACK_RATIOS) *
        config.NUM_RUNS
    )
    
    experiment_count = 0
    
    for algorithm_name in config.ALGORITHMS:
        print(f"\n\n{'#'*80}")
        print(f"# ALGORITHM: {algorithm_name}")
        print(f"{'#'*80}\n")
        
        all_results[algorithm_name] = {}
        
        for attack_type in config.ATTACK_TYPES:
            attack_key = attack_type or 'clean'
            all_results[algorithm_name][attack_key] = {}
            
            for attack_ratio in config.ATTACK_RATIOS:
                # Skip non-zero ratios for clean setting
                if attack_type is None and attack_ratio > 0:
                    continue
                
                ratio_key = f"ratio_{attack_ratio}"
                all_results[algorithm_name][attack_key][ratio_key] = []
                
                for run_id in range(config.NUM_RUNS):
                    experiment_count += 1
                    
                    print(f"\n[Experiment {experiment_count}/{total_experiments}]")
                    
                    # Set random seed for reproducibility
                    np.random.seed(config.RANDOM_SEED + run_id)
                    tf.random.set_seed(config.RANDOM_SEED + run_id)

                    # Create per-run output directory
                    run_output_dir = f"{config.BASE_OUTPUT_DIR}/{algorithm_name}_{config.DATASET}_RUN{run_id+1}"
                    os.makedirs(run_output_dir, exist_ok=True)
                    os.makedirs(os.path.join(run_output_dir, 'plots'), exist_ok=True)
                    os.makedirs(os.path.join(run_output_dir, 'data'), exist_ok=True)

                    # Update config to use this directory for this run
                    config.OUTPUT_DIR = run_output_dir

                    print(f"   Output directory: {run_output_dir}")

                    # Create algorithm instance
                    if algorithm_name == 'VFL_TS':
                        algo = VFLTSAlgorithm(input_shape, num_classes, shared_val_set)
                    elif algorithm_name == 'FedAvg':
                        algo = FedAvgAlgorithm(input_shape, num_classes)
                    elif algorithm_name == 'FedProx':
                        algo = FedProxAlgorithm(input_shape, num_classes, mu=0.01)
                    elif algorithm_name == 'TrimmedMean':
                        algo = TrimmedMeanAlgorithm(input_shape, num_classes, trim_ratio=0.2)
                    elif algorithm_name == 'Krum':
                        algo = KrumAlgorithm(input_shape, num_classes, n_byzantine=None, multi_krum=False)
                    elif algorithm_name == 'FLTrust':
                        algo = FLTrustAlgorithm(input_shape, num_classes, root_dataset)
                    else:
                        print(f"⚠️  Algorithm {algorithm_name} not implemented, skipping")
                        continue
                    
                    # Run training
                    history = run_federated_training(
                        algo, client_data, X_test, y_test,
                        attack_type, attack_ratio, run_id
                    )

                    # Accumulate into all_results (used by generate_all_plots later)
                    all_results[algorithm_name][attack_key][ratio_key].append(history)

                    # Accumulate into per_run_results for per-folder saving
                    run_key = (algorithm_name, run_id)
                    if run_key not in per_run_results:
                        per_run_results[run_key] = {}
                    if attack_key not in per_run_results[run_key]:
                        per_run_results[run_key][attack_key] = {}
                    per_run_results[run_key][attack_key][ratio_key] = [history]

        # ------------------------------------------------------------
        # Write this algorithm's run folders immediately (crash-safe)
        # ------------------------------------------------------------
        for (algo_name, run_id), attack_data in per_run_results.items():
            if algo_name != algorithm_name:
                continue

            run_output_dir = f"{config.BASE_OUTPUT_DIR}/{algo_name}_{config.DATASET}_RUN{run_id+1}"
            os.makedirs(os.path.join(run_output_dir, 'data'), exist_ok=True)
            os.makedirs(os.path.join(run_output_dir, 'plots'), exist_ok=True)

            # { algorithm_name: { attack_key: { ratio_key: [history] } } }
            run_nested = {algo_name: attack_data}

            # --- all_results.txt  (pprint, matches original format) ---
            with open(os.path.join(run_output_dir, 'data', 'all_results.txt'), 'w', encoding='utf-8') as f:
                pprint.pprint(run_nested, stream=f, width=200)

            # --- run_results.pkl ---
            with open(os.path.join(run_output_dir, 'data', 'run_results.pkl'), 'wb') as f:
                pickle.dump(run_nested, f)

            # --- run_results.json ---
            with open(os.path.join(run_output_dir, 'data', 'run_results.json'), 'w', encoding='utf-8') as f:
                json.dump(to_json_serializable(run_nested), f, indent=2)

            # --- run_summary.txt ---
            with open(os.path.join(run_output_dir, 'run_summary.txt'), 'w', encoding='utf-8') as f:
                f.write(f"Algorithm : {algo_name}\n")
                f.write(f"Dataset   : {config.DATASET}\n")
                f.write(f"Run       : {run_id + 1}\n")
                f.write(f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*60}\n\n")
                for atk_key, ratios in attack_data.items():
                    f.write(f"Attack: {atk_key}\n")
                    for ratio_key, history_list in ratios.items():
                        history = history_list[0]  # single run per folder
                        final_acc = history.get('accuracy', [None])[-1]
                        final_loss = history.get('loss', [None])[-1]
                        f.write(f"  {ratio_key}  ->  acc={final_acc}  loss={final_loss}\n")
                    f.write("\n")

            print(f"   Saved results: {run_output_dir}/")

    print(f"\n✅ All experiments complete!")
    print(f"   Results saved in: {config.BASE_OUTPUT_DIR}/")

    return all_results


# ============================================================================
# VISUALIZATION AND ANALYSIS
# ============================================================================

def safe_save_plot(plot_file, filename_display=None, max_retries=3):
    """
    Safely save a plot with error handling for permission issues.

    Args:
        plot_file: Full path to save the plot
        filename_display: Display name for user feedback (optional)
        max_retries: Maximum number of retry attempts
    """
    import time

    if filename_display is None:
        filename_display = os.path.basename(plot_file)

    for attempt in range(max_retries):
        try:
            plt.savefig(plot_file, bbox_inches='tight', dpi=300)
            print(f"   Saved: {filename_display}")
            return True
        except PermissionError as e:
            if attempt < max_retries - 1:
                print(f"   ⚠️  File locked, retrying in 1 second... (attempt {attempt + 1}/{max_retries})")
                time.sleep(1)
            else:
                # Try with timestamp suffix
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                base, ext = os.path.splitext(plot_file)
                new_plot_file = f"{base}_{timestamp}{ext}"
                try:
                    plt.savefig(new_plot_file, bbox_inches='tight', dpi=300)
                    print(f"   ⚠️  Original file locked - saved as: {os.path.basename(new_plot_file)}")
                    return True
                except Exception as e2:
                    print(f"   ❌ Error saving plot: {e2}")
                    return False
        except Exception as e:
            print(f"   ❌ Error saving plot: {e}")
            return False

    return False


def plot_learning_curves(all_results):
    """Plot accuracy curves for ALL algorithms and attack scenarios with ALL ratios."""
    print("\n📊 Generating comprehensive learning curve plots for ALL configurations...")

    plot_count = 0

    # Iterate through ALL attack types
    for attack_type in config.ATTACK_TYPES:
        attack_key = attack_type or 'clean'
        attack_name = attack_type or 'No Attack (Clean)'

        # Get all available ratios for this attack type
        available_ratios = []
        if attack_type is None:
            available_ratios = [0.0]
        else:
            available_ratios = config.ATTACK_RATIOS

        # Create a plot for EACH ratio
        for ratio in available_ratios:
            ratio_key = f"ratio_{ratio}"

            # Check if any algorithm has data for this configuration
            has_data = False
            for algo_name in config.ALGORITHMS:
                if algo_name in all_results and attack_key in all_results[algo_name]:
                    if ratio_key in all_results[algo_name][attack_key]:
                        has_data = True
                        break

            if not has_data:
                continue

            # Create individual figure for this attack × ratio combination
            fig, ax = plt.subplots(figsize=(10, 6))
            title = f'{attack_name} (Malicious Ratio: {ratio*100:.0f}%)'
            fig.suptitle(f'Learning Curves: {title}', fontsize=14, fontweight='bold')

            # Plot each algorithm
            for algo_name in config.ALGORITHMS:
                if algo_name not in all_results:
                    continue

                if attack_key not in all_results[algo_name]:
                    continue

                if ratio_key not in all_results[algo_name][attack_key]:
                    continue

                runs = all_results[algo_name][attack_key][ratio_key]

                if len(runs) == 0:
                    continue

                # Compute mean and std across runs
                accuracies = np.array([run['accuracy'] for run in runs])
                mean_acc = accuracies.mean(axis=0)
                std_acc = accuracies.std(axis=0)

                rounds = np.arange(1, len(mean_acc) + 1)

                # Plot
                ax.plot(rounds, mean_acc, label=algo_name, linewidth=2.5, marker='o', markersize=4)
                ax.fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc,
                               alpha=0.2)

            ax.set_xlabel('Communication Round', fontsize=12)
            ax.set_ylabel('Test Accuracy', fontsize=12)
            ax.legend(fontsize=10, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])

            plt.tight_layout()

            if config.SAVE_PLOTS:
                # Create safe filename
                attack_filename = attack_key.replace(' ', '_').lower()
                plot_file = os.path.join(config.OUTPUT_DIR, 'plots',
                                        f'learning_curve_{attack_filename}_ratio_{ratio}.pdf')
                filename_display = f'learning_curve_{attack_filename}_ratio_{ratio}.pdf'

                if safe_save_plot(plot_file, filename_display):
                    plot_count += 1

            plt.close()

    print(f"   ✅ Generated {plot_count} learning curve plots (one per attack×ratio configuration)")


def plot_robustness_analysis(all_results):
    """Plot robustness under varying attack ratios."""
    print("\n📊 Generating robustness analysis plots...")
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle('VFL-TS: Robustness Under Increasing Attack Intensity',
                 fontsize=14, fontweight='bold')
    
    attack_types = ['byzantine_signflip',
                     'byzantine_gaussian',
                     'byzantine_scaling',
                    'label_flipping']

    attack_titles = ['Byzantine Sign Flip', 
                     'Byzantine Gaussian', 
                     'Byzantine Scaling', 
                     'Label Flipping']
    
    for idx, (attack_type, title) in enumerate(zip(attack_types, attack_titles)):
        ax = axes[idx]
        
        for algo_name in config.ALGORITHMS:
            if algo_name not in all_results:
                continue
            
            if attack_type not in all_results[algo_name]:
                continue
            
            final_accs = []
            ratios = []
            
            for ratio in config.ATTACK_RATIOS:
                if ratio == 0:
                    continue
                
                ratio_key = f"ratio_{ratio}"
                
                if ratio_key not in all_results[algo_name][attack_type]:
                    continue
                
                runs = all_results[algo_name][attack_type][ratio_key]
                
                # Get final accuracy (average over last 5 rounds)
                final_acc = np.mean([np.mean(run['accuracy'][-5:]) for run in runs])
                
                final_accs.append(final_acc)
                ratios.append(ratio * 100)
            
            if final_accs:
                ax.plot(ratios, final_accs, marker='o', label=algo_name, 
                       linewidth=2, markersize=8)
        
        ax.set_xlabel('Malicious Client Ratio (%)')
        ax.set_ylabel('Final Test Accuracy')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    
    plt.tight_layout()

    if config.SAVE_PLOTS:
        plot_file = os.path.join(config.OUTPUT_DIR, 'plots', 'robustness_analysis.pdf')
        safe_save_plot(plot_file, 'robustness_analysis.pdf')

    plt.close()


def plot_reputation_evolution(all_results):
    """Plot reputation evolution for VFL-TS showing honest vs malicious clients for ALL attack scenarios."""
    print("\n📊 Generating comprehensive reputation evolution plots for ALL attack configurations...")

    # Check if VFL-TS results exist
    if 'VFL_TS' not in all_results:
        print("   ⚠️  No VFL-TS results found - skipping reputation plots")
        return

    plot_count = 0

    # Iterate through ALL attack types and ratios
    for attack_type in config.ATTACK_TYPES:
        if attack_type is None:  # Skip clean scenario (no malicious clients)
            continue

        attack_key = attack_type
        attack_name = attack_type.replace('_', ' ').title()

        if attack_key not in all_results['VFL_TS']:
            continue

        # Iterate through all ratios
        for ratio in config.ATTACK_RATIOS:
            ratio_key = f"ratio_{ratio}"

            if ratio_key not in all_results['VFL_TS'][attack_key]:
                continue

            runs = all_results['VFL_TS'][attack_key][ratio_key]

            if not runs or 'reputations' not in runs[0] or len(runs[0]['reputations']) == 0:
                continue

            # Use first run for visualization
            run = runs[0]
            reputations_over_time = np.array(run['reputations'])

            if reputations_over_time.size == 0:
                continue

            num_rounds, num_clients = reputations_over_time.shape

            # Get actual malicious client indices
            if 'malicious_clients' in run and len(run['malicious_clients']) > 0:
                malicious_indices = np.array(run['malicious_clients'])
                honest_indices = np.array([i for i in range(num_clients) if i not in malicious_indices])
            else:
                # Fallback: infer from final reputations
                final_reputations = reputations_over_time[-1, :]
                malicious_threshold = 0.5
                malicious_indices = np.where(final_reputations < malicious_threshold)[0]
                honest_indices = np.where(final_reputations >= malicious_threshold)[0]

            # Create plot
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f'VFL-TS: Reputation Evolution - {attack_name} (Ratio: {ratio*100:.0f}%)',
                         fontsize=14, fontweight='bold')

            rounds = np.arange(1, num_rounds + 1)

            # Plot 1: Individual client reputations
            ax1 = axes[0]

            # Plot honest clients (blue, thin lines)
            for idx in honest_indices[:min(10, len(honest_indices))]:
                ax1.plot(rounds, reputations_over_time[:, idx],
                        color='blue', alpha=0.3, linewidth=1)

            # Plot malicious clients (red, thicker lines)
            for idx in malicious_indices:
                ax1.plot(rounds, reputations_over_time[:, idx],
                        color='red', alpha=0.6, linewidth=2)

            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='blue', linewidth=2, label='Honest Clients'),
                Line2D([0], [0], color='red', linewidth=2, label='Malicious Clients')
            ]
            ax1.legend(handles=legend_elements, loc='best')

            ax1.set_xlabel('Communication Round')
            ax1.set_ylabel('Reputation Score')
            ax1.set_title('Individual Client Reputations')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([0, 1.1])
            ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

            # Plot 2: Average reputations (honest vs malicious)
            ax2 = axes[1]

            if len(honest_indices) > 0:
                honest_mean = reputations_over_time[:, honest_indices].mean(axis=1)
                honest_std = reputations_over_time[:, honest_indices].std(axis=1)
                ax2.plot(rounds, honest_mean, color='blue', linewidth=2.5, label='Honest (avg)')
                ax2.fill_between(rounds, honest_mean - honest_std, honest_mean + honest_std,
                                color='blue', alpha=0.2)

            if len(malicious_indices) > 0:
                malicious_mean = reputations_over_time[:, malicious_indices].mean(axis=1)
                malicious_std = reputations_over_time[:, malicious_indices].std(axis=1)
                ax2.plot(rounds, malicious_mean, color='red', linewidth=2.5, label='Malicious (avg)')
                ax2.fill_between(rounds, malicious_mean - malicious_std, malicious_mean + malicious_std,
                                color='red', alpha=0.2)

            ax2.set_xlabel('Communication Round')
            ax2.set_ylabel('Average Reputation Score')
            ax2.set_title('Average Reputation: Honest vs. Malicious')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([0, 1.1])
            ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

            plt.tight_layout()

            if config.SAVE_PLOTS:
                attack_filename = attack_key.replace(' ', '_').lower()
                plot_file = os.path.join(config.OUTPUT_DIR, 'plots',
                                        f'reputation_evolution_{attack_filename}_ratio_{ratio}.pdf')
                filename_display = f'reputation_evolution_{attack_filename}_ratio_{ratio}.pdf'

                if safe_save_plot(plot_file, filename_display):
                    plot_count += 1

            plt.close()

            # Print statistics for this configuration
            if config.VERBOSE:
                print(f"\n   📊 Reputation Statistics [{attack_name}, Ratio: {ratio*100:.0f}%]:")
                print(f"      Malicious clients: {len(malicious_indices)}/{num_clients}")
                if len(honest_indices) > 0:
                    print(f"      Final honest reputation: {honest_mean[-1]:.3f} ± {honest_std[-1]:.3f}")
                if len(malicious_indices) > 0:
                    print(f"      Final malicious reputation: {malicious_mean[-1]:.3f} ± {malicious_std[-1]:.3f}")
                    print(f"      Reputation decline: {reputations_over_time[0, malicious_indices].mean():.3f} → {malicious_mean[-1]:.3f}")

    print(f"   ✅ Generated {plot_count} reputation evolution plots (one per attack×ratio configuration)")


def plot_comprehensive_summary_heatmaps(all_results):
    """Generate comprehensive heatmap summaries showing performance across ALL configurations."""
    print("\n📊 Generating comprehensive summary heatmaps...")

    if len(config.ALGORITHMS) == 0:
        print("   ⚠️  No algorithms found - skipping summary plots")
        return

    plot_count = 0

    # For each algorithm, create a heatmap showing performance across attack types and ratios
    for algo_name in config.ALGORITHMS:
        if algo_name not in all_results:
            continue

        # Prepare data matrix: rows = attack types, columns = ratios
        attack_types_with_clean = [None] + [att for att in config.ATTACK_TYPES if att is not None]
        attack_labels = ['Clean'] + [att.replace('_', ' ').title() for att in config.ATTACK_TYPES if att is not None]

        # Create matrix
        num_attacks = len(attack_types_with_clean)
        num_ratios = len(config.ATTACK_RATIOS) + 1  # +1 for clean (ratio 0.0)

        # Initialize with NaN
        performance_matrix = np.full((num_attacks, num_ratios), np.nan)
        ratio_labels = ['0%'] + [f'{int(r*100)}%' for r in config.ATTACK_RATIOS]

        # Fill matrix
        for i, attack_type in enumerate(attack_types_with_clean):
            attack_key = attack_type or 'clean'

            if attack_key not in all_results[algo_name]:
                continue

            # For clean, only ratio 0.0
            if attack_type is None:
                ratio_key = 'ratio_0.0'
                if ratio_key in all_results[algo_name][attack_key]:
                    runs = all_results[algo_name][attack_key][ratio_key]
                    final_acc = np.mean([np.mean(run['accuracy'][-5:]) for run in runs])
                    performance_matrix[i, 0] = final_acc
            else:
                # For attack scenarios, fill all ratios
                for j, ratio in enumerate([0.0] + config.ATTACK_RATIOS):
                    ratio_key = f"ratio_{ratio}"
                    if ratio_key in all_results[algo_name][attack_key]:
                        runs = all_results[algo_name][attack_key][ratio_key]
                        final_acc = np.mean([np.mean(run['accuracy'][-5:]) for run in runs])
                        performance_matrix[i, j] = final_acc

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        # Use seaborn for better heatmap
        sns.heatmap(performance_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                   xticklabels=ratio_labels, yticklabels=attack_labels,
                   vmin=0, vmax=1, cbar_kws={'label': 'Test Accuracy'},
                   linewidths=0.5, linecolor='gray', ax=ax)

        ax.set_xlabel('Malicious Client Ratio', fontsize=12)
        ax.set_ylabel('Attack Type', fontsize=12)
        ax.set_title(f'{algo_name}: Performance Across All Configurations', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if config.SAVE_PLOTS:
            plot_file = os.path.join(config.OUTPUT_DIR, 'plots',
                                    f'summary_heatmap_{algo_name}.pdf')
            filename_display = f'summary_heatmap_{algo_name}.pdf'

            if safe_save_plot(plot_file, filename_display):
                plot_count += 1

        plt.close()

    print(f"   ✅ Generated {plot_count} summary heatmap plots")


def plot_algorithm_comparison_grid(all_results):
    """Generate grid comparing all algorithms across different attack scenarios."""
    print("\n📊 Generating algorithm comparison grid...")

    if len(config.ALGORITHMS) < 2:
        print("   ⚠️  Need at least 2 algorithms for comparison - skipping")
        return

    # Create comparison for each attack type
    for attack_type in config.ATTACK_TYPES:
        if attack_type is None:
            continue  # Skip clean for comparison

        attack_key = attack_type
        attack_name = attack_type.replace('_', ' ').title()

        # Check if we have data
        has_data = False
        for algo_name in config.ALGORITHMS:
            if algo_name in all_results and attack_key in all_results[algo_name]:
                has_data = True
                break

        if not has_data:
            continue

        # Create subplots for each ratio
        num_ratios = len(config.ATTACK_RATIOS)
        fig, axes = plt.subplots(1, num_ratios, figsize=(6 * num_ratios, 5))

        if num_ratios == 1:
            axes = [axes]

        fig.suptitle(f'Algorithm Comparison: {attack_name} Attack', fontsize=14, fontweight='bold')

        for idx, ratio in enumerate(config.ATTACK_RATIOS):
            ax = axes[idx]
            ratio_key = f"ratio_{ratio}"

            # Collect data for all algorithms
            algo_names = []
            final_accs = []
            colors = []

            for algo_name in config.ALGORITHMS:
                if algo_name not in all_results:
                    continue

                if attack_key not in all_results[algo_name]:
                    continue

                if ratio_key not in all_results[algo_name][attack_key]:
                    continue

                runs = all_results[algo_name][attack_key][ratio_key]
                final_acc = np.mean([np.mean(run['accuracy'][-5:]) for run in runs])

                algo_names.append(algo_name)
                final_accs.append(final_acc)

                # Color: green for VFL-TS variants, blue for others
                if 'VFL' in algo_name or 'TS' in algo_name:
                    colors.append('forestgreen')
                else:
                    colors.append('steelblue')

            # Create bar plot
            if len(algo_names) > 0:
                bars = ax.bar(algo_names, final_accs, color=colors, alpha=0.7, edgecolor='black')

                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=9)

            ax.set_ylabel('Test Accuracy', fontsize=11)
            ax.set_title(f'Ratio: {ratio*100:.0f}%', fontsize=12)
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if config.SAVE_PLOTS:
            attack_filename = attack_key.replace(' ', '_').lower()
            plot_file = os.path.join(config.OUTPUT_DIR, 'plots',
                                    f'comparison_grid_{attack_filename}.pdf')
            filename_display = f'comparison_grid_{attack_filename}.pdf'

            safe_save_plot(plot_file, filename_display)

        plt.close()


def generate_statistical_report(all_results):
    """Generate statistical analysis report with Wilcoxon tests."""
    print("\n📊 Generating statistical analysis report...")
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("VFL-TS STATISTICAL ANALYSIS REPORT")
    report_lines.append("="*80)
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Dataset: {config.DATASET}")
    report_lines.append(f"Clients: {config.NUM_CLIENTS}, Rounds: {config.NUM_ROUNDS}")
    report_lines.append(f"Runs: {config.NUM_RUNS}\n")
    
    # Compare VFL-TS vs FedAvg for each attack scenario
    if 'VFL_TS' in all_results and 'FedAvg' in all_results:
        report_lines.append("\n" + "="*80)
        report_lines.append("PAIRWISE COMPARISON: VFL-TS vs. FedAvg")
        report_lines.append("="*80)
        
        for attack_type in config.ATTACK_TYPES:
            attack_key = attack_type or 'clean'
            
            if attack_key not in all_results['VFL_TS'] or attack_key not in all_results['FedAvg']:
                continue
            
            report_lines.append(f"\n{'-'*80}")
            report_lines.append(f"Attack: {attack_type or 'None (Clean)'}")
            report_lines.append(f"{'-'*80}")
            
            for ratio in config.ATTACK_RATIOS:
                if attack_type is None and ratio > 0:
                    continue
                
                ratio_key = f"ratio_{ratio}"
                
                if (ratio_key not in all_results['VFL_TS'][attack_key] or
                    ratio_key not in all_results['FedAvg'][attack_key]):
                    continue
                
                # Get final accuracies
                vflts_accs = [np.mean(run['accuracy'][-5:]) 
                             for run in all_results['VFL_TS'][attack_key][ratio_key]]
                fedavg_accs = [np.mean(run['accuracy'][-5:])
                              for run in all_results['FedAvg'][attack_key][ratio_key]]
                
                # Compute statistics
                vflts_mean = np.mean(vflts_accs)
                vflts_std = np.std(vflts_accs)
                fedavg_mean = np.mean(fedavg_accs)
                fedavg_std = np.std(fedavg_accs)
                
                improvement = ((vflts_mean - fedavg_mean) / fedavg_mean) * 100
                
                # Wilcoxon signed-rank test
                if len(vflts_accs) == len(fedavg_accs) and len(vflts_accs) > 1:
                    stat, p_value = wilcoxon(vflts_accs, fedavg_accs)
                    significant = "YES" if p_value < 0.05 else "NO"
                else:
                    p_value = None
                    significant = "N/A"
                
                report_lines.append(f"\nAttack Ratio: {ratio*100:.0f}%")
                report_lines.append(f"  VFL-TS:  {vflts_mean:.4f} +/- {vflts_std:.4f}")
                report_lines.append(f"  FedAvg:  {fedavg_mean:.4f} +/- {fedavg_std:.4f}")
                report_lines.append(f"  Improvement: {improvement:+.2f}%")
                if p_value is not None:
                    report_lines.append(f"  Wilcoxon p-value: {p_value:.4f} (Significant: {significant})")
                
    # Save report with UTF-8 encoding
    report_text = "\n".join(report_lines)
    report_file = os.path.join(config.OUTPUT_DIR, 'statistical_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"   Saved: {report_file}")
    
    # Also print to console
    print("\n" + report_text)


def generate_all_plots(all_results):
    """Generate all plots and analysis."""
    print("\n" + "="*80)
    print("📊 GENERATING PLOTS AND ANALYSIS")
    print("="*80)

    # Write entire results to txt
    output_path = os.path.join(config.OUTPUT_DIR, "all_results.txt")
    with open(output_path, "w") as f:
        pprint.pprint(all_results, stream=f, width=120, sort_dicts=False)

    # Generate comprehensive plots for ALL configurations
    plot_learning_curves(all_results)  # Now generates plots for ALL attack×ratio combinations
    plot_robustness_analysis(all_results)  # Robustness across ratios (existing)
    plot_reputation_evolution(all_results)  # Now generates plots for ALL attack×ratio combinations

    # Generate summary visualizations
    plot_comprehensive_summary_heatmaps(all_results)  # NEW: Heatmaps for each algorithm
    plot_algorithm_comparison_grid(all_results)  # NEW: Bar chart comparisons

    # Generate statistical report
    generate_statistical_report(all_results)

    print("\n✅ All plots and analysis complete!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def load_results_from_file(results_file=None):
    """
    Load previously saved experimental results from pickle file.
    
    This allows you to skip re-running experiments and directly generate
    plots or perform analysis on existing results.
    
    Args:
        results_file: Path to pickle file. If None, uses default location.
    
    Returns:
        all_results: Dictionary containing all experimental results
    
    Raises:
        FileNotFoundError: If results file doesn't exist
    """
    if results_file is None:
        results_file = os.path.join(config.OUTPUT_DIR, 'data', 'all_results.pkl')
    
    if not os.path.exists(results_file):
        raise FileNotFoundError(
            f"Results file not found: {results_file}\n"
            f"Please run experiments first or specify correct file path."
        )
    
    print(f"\n📥 Loading results from: {results_file}")
    
    with open(results_file, 'rb') as f:
        all_results = pickle.load(f)
    
    print(f"✅ Results loaded successfully!")
    
    # Print summary
    print(f"\n📊 Loaded Results Summary:")
    print(f"   Algorithms: {list(all_results.keys())}")
    
    for algo_name, algo_results in all_results.items():
        print(f"\n   {algo_name}:")
        for config_name in algo_results.keys():
            num_runs = len(algo_results[config_name]) if isinstance(algo_results[config_name], list) else 1
            print(f"     - {config_name}: {num_runs} run(s)")
    
    return all_results


def quick_plot_from_file(results_file=None):
    """
    Quick function to load results and generate plots without re-running experiments.
    
    Args:
        results_file: Path to pickle file. If None, uses default location.
    """
    # Load results
    all_results = load_results_from_file(results_file)
    
    # Write everything to txt
    output_path = os.path.join(config.OUTPUT_DIR, "all_results.txt")
    with open(output_path, "w") as f:
        pprint.pprint(all_results, stream=f, width=120, sort_dicts=False)
        
    # Generate plots
    print("\n" + "="*80)
    print("📊 GENERATING PLOTS FROM LOADED RESULTS")
    print("="*80)
    
    generate_all_plots(all_results)
    
    print("\n" + "="*80)
    print("✅ PLOTTING COMPLETE!")
    print("="*80)
    print(f"\nPlots saved to: {config.OUTPUT_DIR}/plots/")
    print(f"Statistical report: {config.OUTPUT_DIR}/statistical_report.txt")
    print("="*80 + "\n")



if __name__ == '__main__':
    print("\n" + "🚀"*40)
    print("VFL-TS: Verifiable Federated Learning with Two-Stage Selection")
    print("Research Implementation for Publication")
    print("COMPREHENSIVE DEBUG VERSION - Byzantine Attack Investigation")
    print("🚀"*40 + "\n")
    
    #quick_plot_from_file()
    #exit()

    # Run all experiments
    all_results = run_all_experiments()

    # Note: Individual run results are saved in separate folders
    # Combined analysis/plotting can be done separately using your own tools

    print("\n" + "="*80)
    print("✅ EXPERIMENT COMPLETE!")
    print("="*80)
    print(f"\nIndividual run results saved to: {config.BASE_OUTPUT_DIR}/")
    print(f"  Folder format: {{Algorithm}}_{{Dataset}}_RUN{{#}}/")
    print(f"  Each folder contains:")
    print(f"    - data/run_results.pkl (pickle format)")
    print(f"    - data/run_results.json (JSON format)")
    print(f"    - run_summary.txt (text summary)")
    print("\n📝 Use your own analysis tools to aggregate results across runs!")
    print("="*80 + "\n")