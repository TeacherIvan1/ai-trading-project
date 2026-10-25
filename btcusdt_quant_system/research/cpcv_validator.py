"""
Combinatorial Purged Cross-Validation, PBO & Deflated Sharpe Ratio
===================================================================
Production research module implementing:
1. Combinatorial Purged & Embargoed Cross-Validation (CPCV) (López de Prado, AFML Ch. 12)
2. Probability of Backtest Overfitting (PBO) (Bailey et al., 2015)
3. Probabilistic Sharpe Ratio (PSR) & Deflated Sharpe Ratio (DSR) (Bailey & López de Prado, 2014)

Designed specifically for non-IID high-frequency financial time series and Triple-Barrier labels.
Includes high-performance vectorized operations with fallback to standard library math/statistics
to ensure zero-dependency execution across environments.
"""

import itertools
import math
import statistics
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None # type: ignore
    HAS_NUMPY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None # type: ignore
    HAS_PANDAS = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    stats = None # type: ignore
    HAS_SCIPY = False


# ==============================================================================
# Mathematical Helpers (Vectorized with exact standard library fallback)
# ==============================================================================

def norm_cdf(x: float) -> float:
    """Standard Normal Cumulative Distribution Function Phi(x)."""
    if HAS_SCIPY:
        return float(stats.norm.cdf(x))
    # Exact calculation using standard library math.erf: Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
    return 0.5 * (1.0 + math.erf(x / 1.4142135623730951))


def norm_ppf(p: float) -> float:
    """
    Standard Normal Percent Point Function / Probit (Inverse CDF) Phi^{-1}(p).
    Acklam's rational approximation (error < 1.15e-9).
    """
    if HAS_SCIPY:
        return float(stats.norm.ppf(p))

    if p <= 0.0 or p >= 1.0:
        if p == 0.0:
            return -float("inf")
        if p == 1.0:
            return float("inf")
        raise ValueError(f"p must be in (0, 1), got {p}")

    # Coefficients in rational approximations
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    q_low = 0.02425
    q_high = 1.0 - q_low

    if p < q_low:
        # Rational approximation for lower region
        q = math.sqrt(-2.0 * math.log(p))
        return (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= q_high:
        # Rational approximation for central region
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    else:
        # Rational approximation for upper region
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def compute_ranks(values: Sequence[float]) -> List[float]:
    """Computes fractional rank data matching scipy.stats.rankdata."""
    if HAS_SCIPY:
        return list(stats.rankdata(values))
    n = len(values)
    sorted_pairs = sorted([(val, idx) for idx, val in enumerate(values)], key=lambda x: x[0])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_pairs[j + 1][0] == sorted_pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_pairs[k][1]] = avg_rank
        i = j + 1
    return ranks


# ==============================================================================
# 1. Combinatorial Purged & Embargoed Cross-Validation (CPCV)
# ==============================================================================

class CombinatorialPurgedKFold:
    """
    Combinatorial Purged K-Fold Cross-Validation.
    
    Splits T observations into N chronological groups.
    Generates C(N, k) train/test combinations.
    Applies strict Purging (removing training labels whose evaluation window overlaps test set)
    and Embargoing (removing training samples immediately following test periods to prevent AR leakage).
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        pct_embargo: float = 0.01,
    ):
        if n_splits <= n_test_splits:
            raise ValueError(f"n_splits ({n_splits}) must be strictly greater than n_test_splits ({n_test_splits})")
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.pct_embargo = pct_embargo

    @property
    def num_combinations(self) -> int:
        return math.comb(self.n_splits, self.n_test_splits)

    def split(
        self,
        X: Sequence[Any],
        pred_times: Optional[Sequence[Any]] = None,
        eval_times: Optional[Sequence[Any]] = None,
    ) -> Generator[Tuple[List[int], List[int]], None, None]:
        """
        Yields (train_indices, test_indices) for each combination.
        
        Parameters:
        -----------
        X: Feature matrix, time series, or indexable sequence.
        pred_times: Sequence of timestamps when each sample's features are observed (t0).
                    If None, assumes uniform integer index 0..T-1.
        eval_times: Sequence of timestamps when each sample's label is realized / exited (t1).
                    Crucial for Triple-Barrier purging.
        """
        n_samples = len(X)
        indices = list(range(n_samples))

        p_times = list(pred_times) if pred_times is not None else indices
        e_times = list(eval_times) if eval_times is not None else indices

        # 1. Partition sample indices into N contiguous chunks
        step = n_samples / self.n_splits
        group_indices = [
            indices[int(round(i * step)) : int(round((i + 1) * step))]
            for i in range(self.n_splits)
        ]

        embargo_bars = int(n_samples * self.pct_embargo)

        # 2. Iterate through all combinations of test groups
        for test_group_keys in itertools.combinations(range(self.n_splits), self.n_test_splits):
            test_idx: List[int] = []
            test_limits: List[Tuple[Any, Any]] = []

            for gk in test_group_keys:
                grp = group_indices[gk]
                if grp:
                    test_idx.extend(grp)
                    test_limits.append((p_times[grp[0]], e_times[grp[-1]]))

            test_set = set(test_idx)
            train_mask = [True] * n_samples

            # A. Exclude test samples from training set
            for tidx in test_idx:
                train_mask[tidx] = False

            # B. Purge: If a training sample's label spans into any test block, purge it
            for test_t0, test_t1 in test_limits:
                for idx in range(n_samples):
                    if not train_mask[idx]:
                        continue
                    # Overlap condition: (sample_t0 <= test_t1) and (sample_t1 >= test_t0)
                    if p_times[idx] <= test_t1 and e_times[idx] >= test_t0:
                        train_mask[idx] = False

                # C. Embargo: Remove training samples immediately following test block
                if embargo_bars > 0:
                    last_test_bar = -1
                    for idx in range(n_samples):
                        if p_times[idx] <= test_t1:
                            last_test_bar = max(last_test_bar, idx)
                    if last_test_bar >= 0:
                        embargo_end_bar = min(n_samples, last_test_bar + embargo_bars + 1)
                        for emb_idx in range(last_test_bar, embargo_end_bar):
                            train_mask[emb_idx] = False

            train_idx = [idx for idx in range(n_samples) if train_mask[idx]]
            yield train_idx, test_idx


# ==============================================================================
# 2. Probability of Backtest Overfitting (PBO)
# ==============================================================================

class ProbabilityOfBacktestOverfitting:
    """
    Computes Probability of Backtest Overfitting (PBO) using the CSCV/CPCV framework
    (Bailey, Borwein, López de Prado, and Zhu, 2015).
    """

    @staticmethod
    def compute(
        performance_matrix: Union[List[List[float]], Any],
    ) -> Dict[str, Union[float, List[float], bool]]:
        """
        Parameters:
        -----------
        performance_matrix: Matrix of shape (C, M)
            C: number of combinatorial test splits (combinations).
            M: number of strategy configurations / hyperparameter trials tested.
            performance_matrix[c][m] is the performance metric (e.g. Sharpe ratio)
            of model m on combination c out-of-sample.

        Returns:
        --------
        Dict containing:
            pbo: Probability of Backtest Overfitting [0.0, 1.0].
            relative_ranks: List of relative ranks omega_c.
            logits: List of logit values lambda_c.
            mean_oos_rank: Mean relative rank of in-sample optimal model out-of-sample.
            is_overfitted: True if pbo > 0.5.
        """
        # Convert to standard 2D list of floats
        if HAS_NUMPY and isinstance(performance_matrix, np.ndarray):
            matrix = performance_matrix.tolist()
        else:
            matrix = [list(row) for row in performance_matrix]

        C = len(matrix)
        if C == 0:
            raise ValueError("Performance matrix must have at least 1 combination.")
        M = len(matrix[0])
        if M < 2:
            raise ValueError("Need at least 2 strategy configurations to evaluate backtest overfitting.")

        relative_ranks: List[float] = []
        logits: List[float] = []

        # For each combination c, find the best model in-sample (mean across all other splits)
        for c in range(C):
            is_splits = [k for k in range(C) if k != c]
            # Mean per model across in-sample splits
            is_performance = [
                sum(matrix[k][m] for k in is_splits) / len(is_splits)
                for m in range(M)
            ]

            # In-sample optimal model index
            best_is_model_idx = max(range(M), key=lambda m: is_performance[m])

            # Out-of-sample performance for all models on test combination c
            oos_performance = matrix[c]

            # Compute relative rank of best_is_model_idx out-of-sample
            ranks = compute_ranks(oos_performance)
            omega_c = ranks[best_is_model_idx] / (M + 1.0)
            relative_ranks.append(omega_c)

            # Logit transformation with numerical clipping
            omega_clipped = max(1e-6, min(1.0 - 1e-6, omega_c))
            lambda_c = math.log(omega_clipped / (1.0 - omega_clipped))
            logits.append(lambda_c)

        # PBO is proportion of combinations where relative rank <= 0.5
        pbo = sum(1.0 for r in relative_ranks if r <= 0.5) / float(C)
        mean_rank = sum(relative_ranks) / float(len(relative_ranks))

        return {
            "pbo": pbo,
            "relative_ranks": relative_ranks,
            "logits": logits,
            "mean_oos_rank": mean_rank,
            "is_overfitted": bool(pbo > 0.5),
        }


# ==============================================================================
# 3. Probabilistic & Deflated Sharpe Ratio (PSR & DSR)
# ==============================================================================

class DeflatedSharpeRatio:
    """
    Computes Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR)
    (Bailey & López de Prado, 2014; López de Prado, 2018).
    
    Adjusts the estimated Sharpe Ratio for:
    - Skewness and Kurtosis (fat tails and non-normality)
    - Sample length (T)
    - Multiple testing selection bias across N trials
    """

    @staticmethod
    def estimated_sr_std(
        sr: float,
        n_samples: int,
        skew: float = 0.0,
        kurtosis: float = 3.0,
    ) -> float:
        """
        Computes standard error of Sharpe Ratio estimate under non-normality:
        sigma_SR = sqrt( (1 - gamma_3 * SR + ((gamma_4 - 1)/4) * SR^2) / (T - 1) )
        """
        if n_samples <= 1:
            return 1.0
        var = (1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * (sr ** 2)) / (n_samples - 1.0)
        return math.sqrt(max(var, 1e-12))

    @classmethod
    def probabilistic_sharpe_ratio(
        cls,
        observed_sr: float,
        benchmark_sr: float = 0.0,
        n_samples: int = 1000,
        skew: float = 0.0,
        kurtosis: float = 3.0,
    ) -> float:
        """
        Computes PSR: Probability that true Sharpe ratio exceeds benchmark_sr.
        PSR(SR*) = Phi( (SR - SR*) / sigma_SR )
        """
        sr_std = cls.estimated_sr_std(observed_sr, n_samples, skew, kurtosis)
        z = (observed_sr - benchmark_sr) / sr_std
        return norm_cdf(z)

    @classmethod
    def expected_maximum_sr(
        cls,
        n_trials: int,
        mean_trial_sr: float = 0.0,
        std_trial_sr: float = 1.0,
    ) -> float:
        """
        Approximates expected maximum Sharpe ratio under null hypothesis across N independent trials
        using Extreme Value Theory (Euler-Mascheroni approximation):
        E[max_N] approx ( (1 - gamma)*Phi^{-1}(1 - 1/N) + gamma*Phi^{-1}(1 - 1/(N*e)) ) * sigma_trials
        """
        if n_trials <= 1:
            return mean_trial_sr

        euler_mascheroni = 0.5772156649015328606
        p1 = 1.0 - (1.0 / n_trials)
        p2 = 1.0 - (1.0 / (n_trials * math.e))

        p1_clipped = max(1e-12, min(1.0 - 1e-12, p1))
        p2_clipped = max(1e-12, min(1.0 - 1e-12, p2))

        z1 = norm_ppf(p1_clipped)
        z2 = norm_ppf(p2_clipped)

        expected_max_z = (1.0 - euler_mascheroni) * z1 + euler_mascheroni * z2
        return float(mean_trial_sr + std_trial_sr * expected_max_z)

    @classmethod
    def compute(
        cls,
        observed_sr: float,
        all_trials_sr: Sequence[float],
        n_samples: int,
        skew: float = 0.0,
        kurtosis: float = 3.0,
    ) -> Dict[str, Union[float, int, bool]]:
        """
        Computes Deflated Sharpe Ratio (DSR).
        
        Parameters:
        -----------
        observed_sr: The best Sharpe ratio selected among candidate strategies.
        all_trials_sr: Sequence of Sharpe ratios from all N attempted strategy configurations.
        n_samples: Number of returns / bar observations (T).
        skew: Sample return skewness.
        kurtosis: Sample return kurtosis (Pearson, normal = 3.0).
        """
        trials = list(all_trials_sr)
        n_trials = len(trials)
        if n_trials > 1:
            std_trials = statistics.stdev(trials)
            mean_trials = statistics.mean(trials)
        else:
            std_trials = 0.0
            mean_trials = trials[0] if trials else 0.0

        # Expected maximum under selection bias
        expected_max = cls.expected_maximum_sr(n_trials, mean_trials, std_trials)

        # Deflated Sharpe Ratio is PSR evaluated against expected_max as benchmark
        dsr = cls.probabilistic_sharpe_ratio(
            observed_sr=observed_sr,
            benchmark_sr=expected_max,
            n_samples=n_samples,
            skew=skew,
            kurtosis=kurtosis,
        )

        psr_zero = cls.probabilistic_sharpe_ratio(
            observed_sr=observed_sr,
            benchmark_sr=0.0,
            n_samples=n_samples,
            skew=skew,
            kurtosis=kurtosis,
        )

        return {
            "observed_sr": float(observed_sr),
            "expected_max_sr": float(expected_max),
            "deflated_sharpe_ratio": float(dsr),
            "probabilistic_sharpe_ratio_vs_0": float(psr_zero),
            "num_trials": n_trials,
            "std_of_trials": float(std_trials),
            "is_statistically_significant": bool(dsr >= 0.95),
        }


# ==============================================================================
# Comprehensive Demonstration & Validation Test
# ==============================================================================
def _demo():
    import random

    print("=" * 70)
    print("CPCV, PBO & Deflated Sharpe Ratio Validation Engine")
    print("=" * 70)

    # 1. CPCV Test with simulated Triple-Barrier event horizons
    n_bars = 1000
    timestamps = [f"2026-01-01 {i // 4:02d}:{(i % 4) * 15:02d}" for i in range(n_bars)]
    
    # Simulate label exit horizons (holding period 5 to 15 bars)
    random.seed(42)
    durations = [random.randint(5, 15) for _ in range(n_bars)]
    eval_timestamps = [timestamps[min(n_bars - 1, i + durations[i])] for i in range(n_bars)]
    pred_timestamps = list(timestamps)

    cpcv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2, pct_embargo=0.01)
    print(f"CPCV Setup: {cpcv.n_splits} splits, {cpcv.n_test_splits} test groups -> {cpcv.num_combinations} paths.")

    splits = list(cpcv.split(timestamps, pred_times=pred_timestamps, eval_times=eval_timestamps))
    print(f"Generated {len(splits)} train/test split combinations successfully.")

    for i, (train_idx, test_idx) in enumerate(splits[:3]):
        train_ratio = len(train_idx) / n_bars
        test_ratio = len(test_idx) / n_bars
        purged_ratio = 1.0 - train_ratio - test_ratio
        print(f"  Split {i+1:02d}: Train={train_ratio*100:.1f}%, Test={test_ratio*100:.1f}%, Purged/Embargoed={purged_ratio*100:.1f}%")

    # 2. PBO Test with synthetic strategy trials
    n_combinations = cpcv.num_combinations
    n_configs = 25
    # Synthetic performance matrix with random noise (simulating multiple testing)
    perf_matrix = [
        [random.gauss(0.05, 0.3) for _ in range(n_configs)]
        for _ in range(n_combinations)
    ]
    pbo_result = ProbabilityOfBacktestOverfitting.compute(perf_matrix)
    print(f"\nPBO Evaluation:")
    print(f"  Probability of Backtest Overfitting (PBO): {pbo_result['pbo']:.2%}")
    print(f"  Mean Out-of-Sample Relative Rank: {pbo_result['mean_oos_rank']:.3f}")

    # 3. Deflated Sharpe Ratio Test
    n_samples = 2500  # ~2500 bars
    trial_srs = [random.gauss(0.6, 0.4) for _ in range(50)]  # 50 candidate strategy trials
    best_sr = max(trial_srs)

    dsr_result = DeflatedSharpeRatio.compute(
        observed_sr=best_sr,
        all_trials_sr=trial_srs,
        n_samples=n_samples,
        skew=-0.2,
        kurtosis=4.5,
    )
    print(f"\nDeflated Sharpe Ratio (DSR) Evaluation:")
    print(f"  Observed Best SR: {dsr_result['observed_sr']:.3f}")
    print(f"  Expected Max SR (null hypothesis, {dsr_result['num_trials']} trials): {dsr_result['expected_max_sr']:.3f}")
    print(f"  Deflated Sharpe Ratio (DSR): {dsr_result['deflated_sharpe_ratio']:.2%}")
    print(f"  Conventional PSR (vs SR=0): {dsr_result['probabilistic_sharpe_ratio_vs_0']:.2%}")
    print(f"  Statistically Significant at 95% confidence: {dsr_result['is_statistically_significant']}")
    print("=" * 70)


if __name__ == "__main__":
    _demo()
