"""
scripts/run_pipeline_evaluation.py
==================================
6-row evaluation table testing:
- mode in ['IGNITION', 'REVERSION']
- cusum_mult in [1.1, 1.35]
- Meta-Label probability gates: p >= 0.50 (Unfiltered Base), p >= 0.53, p >= 0.56
- Symmetric barriers: pt_mult=1.5, sl_mult=1.5
- 10 features: ['vpin', 'vpin_cdf', 'vpin_slope', 'kyle_lambda_z', 'ffd_close', 
                'rolling_vol', 'vol_regime_ratio', 'flow_alignment', 'abs_z_dist', 'ret_autocorr']
- Single RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=15, class_weight='balanced_subsample')
- 6 Purged CV folds with 1% embargo
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from core.features_labels import (
    generate_primary_signals,
    apply_triple_barrier_meta_labels,
)
from research.cpcv_validator import DeflatedSharpeRatio, ProbabilityOfBacktestOverfitting


def compute_max_drawdown(rets_series: np.ndarray) -> float:
    if len(rets_series) == 0:
        return 0.0
    cum = np.cumprod(1.0 + rets_series)
    peaks = np.maximum.accumulate(cum)
    dd = (peaks - cum) / peaks
    return float(np.max(dd))


def run_6row_evaluation():
    parquet_path = os.path.join(PROJECT_ROOT, "data", "btcusdt_dollar_bars_20m_30d.parquet")
    csv_path = os.path.join(PROJECT_ROOT, "data", "btcusdt_dollar_bars_20m_30d.csv")

    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        from scripts.reaggregate_20m_bars import reaggregate_20m
        df = reaggregate_20m()

    feature_cols = [
        "vpin",
        "vpin_cdf",
        "vpin_slope",
        "kyle_lambda_z",
        "ffd_close",
        "rolling_vol",
        "vol_regime_ratio",
        "flow_alignment",
        "abs_z_dist",
        "ret_autocorr",
    ]

    # Configurations corresponding to the 6 comparison rows:
    # 3 for IGNITION, 3 for REVERSION covering cusum_mult in [1.1, 1.35] and gates [0.50, 0.53, 0.56]
    test_configs = [
        {"mode": "IGNITION", "cusum_mult": 1.10, "gate": 0.50, "label": "p>=0.50 (Base)"},
        {"mode": "IGNITION", "cusum_mult": 1.10, "gate": 0.53, "label": "p>=0.53"},
        {"mode": "IGNITION", "cusum_mult": 1.35, "gate": 0.56, "label": "p>=0.56"},
        {"mode": "REVERSION", "cusum_mult": 1.10, "gate": 0.50, "label": "p>=0.50 (Base)"},
        {"mode": "REVERSION", "cusum_mult": 1.10, "gate": 0.53, "label": "p>=0.53"},
        {"mode": "REVERSION", "cusum_mult": 1.35, "gate": 0.56, "label": "p>=0.56"},
    ]

    # Cache precomputed OOF results for unique (mode, cusum_mult) pairs
    oof_cache = {}
    unique_pairs = {(cfg["mode"], cfg["cusum_mult"]) for cfg in test_configs}

    n_splits = 6

    for mode, cm in sorted(unique_pairs):
        out = generate_primary_signals(df, span=40, cusum_mult=cm, mode=mode)
        meta_labels, rets, exits = apply_triple_barrier_meta_labels(
            out["close"].values,
            out["high"].values,
            out["low"].values,
            out["primary_side"].values,
            out["rolling_vol"].values,
            pt_mult=1.5,
            sl_mult=1.5,
            max_horizon_bars=25,
            roundtrip_fee=0.00025,
        )
        out["meta_label"] = meta_labels
        out["realized_return"] = rets
        out["exit_index"] = exits

        valid = out[feature_cols].notna().all(axis=1)
        df_clean = out[valid].copy().reset_index(drop=True)

        n_samples = len(df_clean)
        indices = np.arange(n_samples)
        split_bounds = np.linspace(0, n_samples, n_splits + 1, dtype=int)
        embargo_bars = max(1, int(n_samples * 0.01))

        oof_probs = []
        oof_rets = []
        oof_folds = []

        for k in range(n_splits):
            test_start = split_bounds[k]
            test_end = split_bounds[k + 1]
            test_idx = indices[test_start:test_end]

            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_start:test_end] = False

            # Purging
            for i in range(n_samples):
                if not train_mask[i]:
                    continue
                if df_clean["primary_side"].iloc[i] != 0:
                    t_exit = df_clean["exit_index"].iloc[i]
                    if max(i, test_start) <= min(t_exit, test_end):
                        train_mask[i] = False

            # 1% Embargo
            embargo_end = min(n_samples, test_end + embargo_bars)
            train_mask[test_end:embargo_end] = False

            train_trades = train_mask & (df_clean["primary_side"] != 0) & (df_clean["meta_label"] >= 0)
            X_tr = df_clean.loc[train_trades, feature_cols].values
            y_tr = df_clean.loc[train_trades, "meta_label"].values.astype(int)

            test_trades = (df_clean.index.isin(test_idx)) & (df_clean["primary_side"] != 0) & (df_clean["meta_label"] >= 0)
            test_trade_idx = df_clean.index[test_trades]

            if len(X_tr) == 0 or len(test_trade_idx) == 0 or len(np.unique(y_tr)) < 2:
                continue

            rf = RandomForestClassifier(
                n_estimators=300,
                max_depth=4,
                min_samples_leaf=15,
                class_weight="balanced_subsample",
                random_state=42,
            )
            rf.fit(X_tr, y_tr)

            X_te = df_clean.loc[test_trade_idx, feature_cols].values
            probs = rf.predict_proba(X_te)[:, 1]
            rets_te = df_clean.loc[test_trade_idx, "realized_return"].values

            oof_probs.extend(probs)
            oof_rets.extend(rets_te)
            oof_folds.extend([k] * len(rets_te))

        oof_cache[(mode, cm)] = {
            "probs": np.array(oof_probs),
            "rets": np.array(oof_rets),
            "folds": np.array(oof_folds),
        }

    # Evaluate each of the 6 comparison rows
    row_records = []
    fold_sr_matrix = np.zeros((n_splits, len(test_configs)))

    for idx, cfg in enumerate(test_configs):
        cached = oof_cache[(cfg["mode"], cfg["cusum_mult"])]
        probs = cached["probs"]
        rets = cached["rets"]
        folds = cached["folds"]

        gate = cfg["gate"]
        mask = probs >= gate
        exec_rets = rets[mask]

        n_trades = len(exec_rets)
        win_rate = (np.mean(exec_rets > 0) * 100.0) if n_trades > 0 else 0.0

        pos_sum = np.sum(exec_rets[exec_rets > 0])
        neg_sum = abs(np.sum(exec_rets[exec_rets < 0]))
        pf_val = float(pos_sum / (neg_sum + 1e-12)) if neg_sum > 0 else 99.0

        if n_trades > 1 and np.std(exec_rets) > 0:
            sr_per_trade = float(np.mean(exec_rets)) / max(float(np.std(exec_rets)), 0.0010)
            sr_ann = sr_per_trade * np.sqrt(n_trades * (365.25 / 30.0))
        else:
            sr_per_trade = 0.0
            sr_ann = 0.0

        mdd = compute_max_drawdown(exec_rets) * 100.0 if n_trades > 0 else 0.0

        # Fold Sharpe for PBO
        for k in range(n_splits):
            fold_mask = mask & (folds == k)
            f_rets = rets[fold_mask]
            if len(f_rets) > 1 and np.std(f_rets) > 0:
                fold_sr_matrix[k, idx] = np.mean(f_rets) / max(float(np.std(f_rets)), 0.0010)
            else:
                fold_sr_matrix[k, idx] = 0.0

        row_records.append({
            "mode": cfg["mode"],
            "cusum_mult": cfg["cusum_mult"],
            "gate_label": cfg["label"],
            "trades": n_trades,
            "win_rate": win_rate,
            "profit_factor": pf_val,
            "sr_ann": sr_ann,
            "sr_per_trade": sr_per_trade,
            "mdd": mdd,
            "exec_rets": exec_rets,
        })

    # Compute PBO across the 6 configurations
    pbo_res = ProbabilityOfBacktestOverfitting.compute(fold_sr_matrix)
    overall_pbo = pbo_res["pbo"] * 100.0

    # Compute DSR for each trial
    all_srs = [r["sr_per_trade"] for r in row_records]

    for r in row_records:
        rets = r["exec_rets"]
        skew = float(stats.skew(rets)) if len(rets) > 2 else 0.0
        kurt = float(stats.kurtosis(rets, fisher=False)) if len(rets) > 2 else 3.0
        dsr_res = DeflatedSharpeRatio.compute(
            observed_sr=r["sr_per_trade"],
            all_trials_sr=all_srs,
            n_samples=max(r["trades"], 2),
            skew=skew,
            kurtosis=kurt,
        )
        r["dsr"] = dsr_res["deflated_sharpe_ratio"] * 100.0
        r["pbo"] = overall_pbo

    # Print clean formatted terminal table
    header = (
        f"{'Row':<4} | {'Mode':<9} | {'CUSUM':<6} | {'Gate':<16} | {'Trades':<6} | "
        f"{'Win Rate':<8} | {'PF':<5} | {'Ann. Sharpe':<11} | {'Max DD':<8} | "
        f"{'PBO':<6} | {'DSR':<6}"
    )
    sep = "=" * len(header)
    print(sep)
    print("META-LABELING 6-ROW STRATEGY COMPARISON: PURGED CV ON $20M DOLLAR BARS")
    print(sep)
    print(header)
    print("-" * len(header))

    for idx, r in enumerate(row_records, 1):
        pf_str = f"{r['profit_factor']:>5.2f}"
        print(
            f"{idx:<4} | {r['mode']:<9} | {r['cusum_mult']:<6.2f} | {r['gate_label']:<16} | "
            f"{r['trades']:<6} | {r['win_rate']:>7.1f}% | {pf_str:<5} | {r['sr_ann']:>11.2f} | "
            f"{r['mdd']:>7.2f}% | {r['pbo']:>5.1f}% | {r['dsr']:>5.1f}%"
        )
    print(sep)


if __name__ == "__main__":
    run_6row_evaluation()
