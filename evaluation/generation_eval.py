from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib.ticker import FormatStrFormatter
from scipy.spatial.distance import jensenshannon


def compare_generated_to_original(
    original_csv: str,
    generated_dir: str,
    output_metrics: str = "./metrics",
    output_plots: str = "./plots",
    feature_num: int = 36,
) -> None:
    """Compare generated and original feature distributions donor by donor."""
    os.makedirs(output_metrics, exist_ok=True)
    os.makedirs(output_plots, exist_ok=True)

    orig_df = pd.read_csv(original_csv)
    feat_names = orig_df.columns[1 : 1 + feature_num]
    rows = []

    for filename in os.listdir(generated_dir):
        if not filename.endswith(".npy"):
            continue
        donor_id = filename[:-4]
        gen = np.load(os.path.join(generated_dir, filename))

        sample0 = f"{donor_id}-0"
        orig = orig_df[orig_df["sample"] == sample0].iloc[:, 1 : 1 + feature_num].values
        if orig.shape[0] == 0:
            print(f"Skipped {donor_id} because {sample0} was not found in the original CSV.")
            continue

        record = {"donor_ID": donor_id}
        for i, feature_name in enumerate(feat_names):
            original_values = orig[:, i]
            generated_values = gen[:, i]

            wd = stats.wasserstein_distance(original_values, generated_values)
            kl = stats.entropy(
                np.histogram(original_values, bins=100, density=True)[0] + 1e-8,
                np.histogram(generated_values, bins=100, density=True)[0] + 1e-8,
            )
            js = jensenshannon(
                np.histogram(original_values, bins=100, density=True)[0],
                np.histogram(generated_values, bins=100, density=True)[0],
            ) ** 2

            record[f"{feature_name}_wd"] = wd
            record[f"{feature_name}_kl"] = kl
            record[f"{feature_name}_js"] = js

            plt.figure(figsize=(4, 3))
            plt.hist(original_values, bins=100, alpha=0.5, density=True, label="original")
            plt.hist(generated_values, bins=100, alpha=0.5, density=True, label="generated")
            plt.title(f"{donor_id} | {feature_name}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_plots, f"{donor_id}_{feature_name}.png"))
            plt.close()

        rows.append(record)

    pd.DataFrame(rows).to_csv(os.path.join(output_metrics, "metrics.csv"), index=False)
    print("Saved donor level metrics and histograms.")


def kl_gaussian(p_vals: np.ndarray, q_vals: np.ndarray, eps: float = 1e-8) -> float:
    """Compute the closed form KL divergence between two univariate Gaussians."""
    mu_p = np.mean(p_vals)
    mu_q = np.mean(q_vals)
    var_p = np.var(p_vals) + eps
    var_q = np.var(q_vals) + eps
    return np.log(np.sqrt(var_q) / np.sqrt(var_p)) + (var_p + (mu_p - mu_q) ** 2) / (2 * var_q) - 0.5


def plot_feature_histogram_by_label(
    generated_dir: str,
    original_csv: str,
    label_xlsx: str,
    donor_ids: list[str],
    output_dir: str = "./feat_by_label_hist",
    feature_num: int = 36,
    bins: int = 50,
    alpha: float = 0.5,
) -> None:
    """Plot generated and original feature histograms grouped by clinical label."""
    os.makedirs(output_dir, exist_ok=True)

    orig_df = pd.read_csv(original_csv)
    labels_df = pd.read_excel(label_xlsx)
    feat_names = orig_df.columns[1 : 1 + feature_num]

    test_labels = labels_df[labels_df["donor_ID"].isin(donor_ids)]
    groups = test_labels.groupby("clinical_diagnosis")["donor_ID"].apply(list).to_dict()

    kl_records = []
    for feature_name in feat_names:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        generated_group_data: dict[str, np.ndarray] = {}
        original_group_data: dict[str, np.ndarray] = {}

        for label_name, ids in groups.items():
            generated_values = []
            for donor_id in ids:
                file_path = os.path.join(generated_dir, f"{donor_id}.npy")
                if not os.path.exists(file_path):
                    continue
                arr = np.load(file_path)[:, feat_names.get_loc(feature_name)].astype(np.float64, copy=False)
                generated_values.append(arr)
            if generated_values:
                values = np.concatenate(generated_values)
                generated_group_data[label_name] = values
                axes[0].hist(values, bins=bins, density=True, alpha=alpha, label=label_name, edgecolor="black")

        axes[0].set_title(f"Generated | {feature_name}")
        axes[0].set_xlabel("Expression value")
        axes[0].set_ylabel("Density")
        axes[0].legend()

        for label_name, ids in groups.items():
            original_values = []
            for donor_id in ids:
                sample0 = f"{donor_id}-0"
                values = orig_df.loc[orig_df["sample"] == sample0, feature_name].values
                if values.size:
                    original_values.append(values.astype(np.float64, copy=False))
            if original_values:
                values = np.concatenate(original_values)
                original_group_data[label_name] = values
                axes[1].hist(values, bins=bins, density=True, alpha=alpha, label=label_name, edgecolor="black")

        axes[1].set_title(f"Original | {feature_name}")
        axes[1].set_xlabel("Expression value")
        axes[1].set_ylabel("Density")
        axes[1].legend()

        for ax in axes:
            ax.yaxis.set_ticks_position("left")
            ax.tick_params(axis="y", which="both", right=False, labelright=False)
            ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{feature_name}.png"))
        plt.close(fig)

        record = {"feature": feature_name}
        all_gen = np.concatenate(list(generated_group_data.values())) if generated_group_data else np.array([])
        all_orig = np.concatenate(list(original_group_data.values())) if original_group_data else np.array([])
        record["KL_overall"] = kl_gaussian(all_orig, all_gen) if all_gen.size and all_orig.size else np.nan

        for label_name in groups.keys():
            if label_name in generated_group_data and label_name in original_group_data:
                record[f"KL_{label_name}"] = kl_gaussian(original_group_data[label_name], generated_group_data[label_name])
            else:
                record[f"KL_{label_name}"] = np.nan
        kl_records.append(record)

    pd.DataFrame(kl_records).to_csv(os.path.join(output_dir, "kl_divergences.csv"), index=False)
    print("Saved grouped histograms and KL divergence summary.")


__all__ = [
    "compare_generated_to_original",
    "plot_feature_histogram_by_label",
    "kl_gaussian",
]
