"""
main_analysis.py

Performs the full evaluation pipeline for motion blur quality analysis.

Steps included:
- Load and merge human blur labels with OFIQ sharpness scores
- Compute inter-rater agreement (Cohen's kappa, Fleiss' kappa)
- Compute correlation between OFIQ scores and human labels
- Generate confusion matrix and classification report
- Create performance plots: EDC (OFIQ + Human), ROC, and DET curves
- Identify failure cases where OFIQ scores disagree with human perception
- Visualize rating distributions and inter-rater agreement heatmap
- Generate contact sheet of failure case images

Inputs:
- image_ratings.xlsx : Human rater annotations (blur levels)
- ofiqlabels.csv      : OFIQ scores and label bins per image

Outputs:
- figures/            : Directory containing PNG plots and contact sheet
- failure_cases.csv   : List of human-blurred but OFIQ-sharp image samples

Dependencies:
- numpy, pandas, matplotlib, seaborn, Pillow
- metrics.py (custom module with plotting and analysis utilities)
"""


import os
import pandas as pd
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import metrics

# Setup
metrics.ensure_figures_folder()

# ----------------------
# Load and prepare data
# ----------------------

# Load human labels
human_labels = pd.read_excel("image_ratings.xlsx")
human_labels.columns = ["Image Name", "Folder", "rater1", "rater2", "rater3"]
human_labels["mean_human"] = human_labels[["rater1", "rater2", "rater3"]].mean(axis=1).round().astype(int)

# Load OFIQ scores
ofiq = pd.read_csv("ofiqlabels.csv", delimiter=";")

# Merge
data = pd.merge(ofiq, human_labels, on=["Image Name", "Folder"])

# Rename for simplicity
data["ofiq_label"] = data["OFIQ Sharpness Label"]

# ----------------------
# Metrics + Reports
# ----------------------

# Cohen’s Kappa
kappas = metrics.compute_cohens_kappas(data)

# Fleiss’ Kappa
fleiss = metrics.compute_fleiss_kappa(data)

# Correlations
correlations = metrics.compute_correlations(data)

# Classification report + Confusion matrix
report_df = metrics.classification_report_df(data["mean_human"], data["ofiq_label"])
print("Classification report:\n", report_df)
metrics.plot_confusion(data)

# Save classification report as image
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')
table = ax.table(
    cellText=report_df.values,
    colLabels=report_df.columns,
    rowLabels=report_df.index,
    loc='center',
    cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.5)
for col in range(len(report_df.columns)):
    if (0, col + 1) in table._cells:
        table[(0, col + 1)].set_facecolor('#deeaf6')
for row in range(len(report_df)):
    if (row + 1, 0) in table._cells:
        table[(row + 1, 0)].set_facecolor('#deeaf6')
plt.title("Classification report", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig("figures/classification_report.png", dpi=300)
plt.close()

# ---------------------
# EDC, ROC, DET Plots
# ---------------------

metrics.plot_edc_curve(data, method="ofiq", save_as="figures/edc_curve_ofiq.png")
metrics.plot_edc_curve(data, method="human", save_as="figures/edc_curve_human.png")
roc_auc = metrics.plot_roc_and_det(data)

# ----------------------------
# Failure cases: Human = 2, OFIQ says sharp
# ----------------------------
failures = data[(data["mean_human"] == 2) & (data["OFIQ Sharpness"] > 0.66)]
failures.to_csv("figures/failure_cases.csv", index=False)

# --------------------------
# Failure image contact sheet
# --------------------------
contact_sheet_rows = 4
contact_sheet_cols = 5
thumb_size = (128, 128)
thumbs = []

fail_samples = failures.head(contact_sheet_rows * contact_sheet_cols)
for _, row in fail_samples.iterrows():
    img_path = os.path.join(row["Folder"], row["Image Name"])
    if os.path.exists(img_path):
        try:
            img = Image.open(img_path).convert("RGB")
            img.thumbnail(thumb_size)
            thumbs.append(img)
        except:
            continue

sheet = Image.new('RGB', (thumb_size[0] * contact_sheet_cols, thumb_size[1] * contact_sheet_rows), color=(255, 255, 255))
for i, thumb in enumerate(thumbs):
    x = i % contact_sheet_cols
    y = i // contact_sheet_cols
    sheet.paste(thumb, (x * thumb_size[0], y * thumb_size[1]))
sheet.save("figures/failure_contact_sheet.jpg")

# ---------------------------
# Human label distribution
# ---------------------------
plt.figure(figsize=(5, 4))
sns.countplot(x='mean_human', data=data, palette='Set2')
plt.title("Distribution of human mean labels")
plt.xlabel("Mean human label")
plt.ylabel("Number of images")
plt.tight_layout()
plt.savefig("figures/human_label_distribution.png", dpi=300)
plt.close()

# -----------------------------
# Inter-rater agreement heatmap
# -----------------------------
disagreements = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        disagreements[i, j] = (data[f"rater{i+1}"] == data[f"rater{j+1}"]).mean()

plt.figure(figsize=(5, 4))
sns.heatmap(disagreements, annot=True, cmap="coolwarm", xticklabels=["r1", "r2", "r3"], yticklabels=["r1", "r2", "r3"])
plt.title("Rater agreement matrix (diagonal = self)")
plt.tight_layout()
plt.savefig("figures/rater_agreement_heatmap.png", dpi=300)
plt.close()

# ---------------------------
# Summary to console
# ---------------------------
print("\nCohen's kappa:")
for k, v in kappas.items():
    print(f"{k}: {v:.2f}")

print(f"\nFleiss' kappa: {fleiss:.2f}")

print("\nCorrelations (Pearson/Spearman):")
for k, v in correlations.items():
    print(f"{k}: {v[0]:.2f} (p={v[1]:.3f})")

print(f"\nROC AUC: {roc_auc:.2f}")

print("\nFigures saved in: figures/")
