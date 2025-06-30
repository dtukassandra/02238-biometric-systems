"""
metrics.py

Utility functions for evaluating agreement and quality in motion blur annotation tasks.

This module includes:
- Inter-rater agreement metrics (Cohen’s kappa, Fleiss’ kappa)
- Correlation analysis (Pearson, Spearman)
- Confusion matrix generation
- ROC and DET curve plotting
- Error vs. Discard (EDC) curves using either human or OFIQ-based quality scores
- Classification report as pandas DataFrame

Outputs:
- PNG figures saved in the 'figures/' directory
- Optional CSV or DataFrame exports handled by calling scripts

Dependencies:
- pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels, scipy
"""


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    cohen_kappa_score,
    roc_curve,
    auc,
    det_curve,
    confusion_matrix,
    classification_report
)
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters


def ensure_figures_folder():
    os.makedirs("figures", exist_ok=True)


def compute_cohens_kappas(df):
    kappas = {
        'rater1_vs_rater2': cohen_kappa_score(df['rater1'], df['rater2']),
        'rater1_vs_rater3': cohen_kappa_score(df['rater1'], df['rater3']),
        'rater2_vs_rater3': cohen_kappa_score(df['rater2'], df['rater3']),
        'mean_vs_ofiq': cohen_kappa_score(df['mean_human'], df['OFIQ Sharpness Label'])
    }
    return kappas


def compute_fleiss_kappa(df):
    ratings = df[['rater1', 'rater2', 'rater3']].to_numpy()
    agg_ratings, _ = aggregate_raters(ratings)
    return fleiss_kappa(agg_ratings)


def compute_correlations(df):
    correlations = {
        'pearson_mean': pearsonr(df['OFIQ Sharpness'], df['mean_human']),
        'spearman_mean': spearmanr(df['OFIQ Sharpness'], df['mean_human']),
        'pearson_rater1': pearsonr(df['OFIQ Sharpness'], df['rater1']),
        'pearson_rater2': pearsonr(df['OFIQ Sharpness'], df['rater2']),
        'pearson_rater3': pearsonr(df['OFIQ Sharpness'], df['rater3']),
    }
    return correlations


def plot_confusion(df, save_as="figures/confusion_matrix.png"):
    cm = confusion_matrix(df['mean_human'], df['OFIQ Sharpness Label'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("OFIQ predicted label")
    plt.ylabel("Human mean label")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(save_as, dpi=300)
    plt.close()


def classification_report_df(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True, labels=[0, 1], zero_division=0)
    return pd.DataFrame(report).transpose().round(2)


def plot_edc_curve(df, method="human", save_as="figures/edc_curve.png"):
    if method == "human":
        sorted_df = df.sort_values(by='mean_human', ascending=False)
    elif method == "ofiq":
        sorted_df = df.sort_values(by='OFIQ Sharpness')
    else:
        raise ValueError("method must be 'human' or 'ofiq'")

    discard_fractions = np.linspace(0, 0.9, 19)
    error_rates = []

    for frac in discard_fractions:
        cutoff = int(len(sorted_df) * frac)
        retained = sorted_df.iloc[cutoff:]
        if len(retained) == 0:
            error_rates.append(np.nan)
            continue
        acc = (retained["mean_human"] == retained["OFIQ Sharpness Label"]).mean()
        error_rates.append(1 - acc)

    plt.figure(figsize=(8, 5))
    plt.plot(discard_fractions * 100, error_rates, marker="o")
    plt.title(f"EDC curve: discard based on {method.capitalize()}", fontsize=14)
    plt.xlabel("Percentage of images discarded")
    plt.ylabel("Classification error (1 - Accuracy)")
    plt.grid(True)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_as, dpi=300)
    plt.close()


def plot_roc_and_det(df, save_roc="figures/roc_ofiq_blur.png", save_det="figures/det_ofiq_blur.png"):
    df['binary_human'] = (df['mean_human'] == 1).astype(int)
    fpr, tpr, _ = roc_curve(df['binary_human'], 1 - df['OFIQ Sharpness'])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve: Blur detection with OFIQ")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_roc, dpi=300)
    plt.close()

    if df['binary_human'].sum() > 0 and df['binary_human'].sum() < len(df):
        fnr, fpr_det, _ = det_curve(df['binary_human'], 1 - df['OFIQ Sharpness'])
        plt.figure(figsize=(6, 5))
        plt.plot(fpr_det, fnr, label="DET Curve")
        plt.xlabel("False match rate (FMR)")
        plt.ylabel("False non-match rate (FNMR)")
        plt.title("DET curve: Blur detection with OFIQ")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_det, dpi=300)
        plt.close()
    else:
        print("⚠️ Cannot plot DET curve: Only one class present.")

    plt.figure(figsize=(6, 5))
    plt.plot(fpr_det, fnr, label="DET Curve")
    plt.xlabel("False match rate (FMR)")
    plt.ylabel("False non-match rate (FNMR)")
    plt.title("DET curve: Blur detection with OFIQ")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_det, dpi=300)
    plt.close()

    return roc_auc
