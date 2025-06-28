"""
analyze_ratings.py

Computes agreement statistics and label distributions for human blur annotations.

Inputs:
- image_ratings.xlsx : Excel file containing image blur labels from multiple raters

Outputs:
- figures/label_distribution.png        : Bar chart showing rating frequency per rater
- tables/rater_agreement_rates.csv      : Pairwise agreement rates between raters (as CSV)
- Console printout of agreement metrics

Requirements:
- pandas, matplotlib
"""


import os
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

# ========== Configuration ==========
file_path = "image_ratings.xlsx"
raters = ["rater1", "rater2", "rater3"]
os.makedirs("figures", exist_ok=True)
os.makedirs("tables", exist_ok=True)

# ========== Load data ==========
df = pd.read_excel(file_path)

# ========== Compute label distributions ==========
label_counts = df[raters].apply(pd.Series.value_counts).fillna(0).astype(int)
print("\nLabel counts per rater:\n", label_counts)

# ========== Compute majority vote ==========
df["majority_vote"] = df[raters].mode(axis=1)[0]

# ========== Pairwise agreement ==========
def pairwise_agreement(col1, col2):
    return (df[col1] == df[col2]).mean()

agreement_results = {
    f"{r1} vs {r2}": pairwise_agreement(r1, r2)
    for r1, r2 in combinations(raters, 2)
}

print("\nAgreement rates between raters:")
for pair, rate in agreement_results.items():
    print(f"{pair}: {rate:.2%}")

# ========== Plot: label distributions ==========
label_counts.plot(kind="bar", figsize=(8, 5))
plt.title("Distribution of blur ratings per rater")
plt.xlabel("Blur level")
plt.ylabel("Number of ratings")
plt.xticks(rotation=0)
plt.legend(title="Rater")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/label_distribution.png")
plt.show()

# ========== Save agreement summary ==========
pd.Series(agreement_results, name="agreement_rate").to_csv("tables/rater_agreement_rates.csv")

print("\nAnalysis complete.")
print(" - Plot saved as: figures/label_distribution.png")
print(" - CSV summary saved as: tables/rater_agreement_rates.csv")
