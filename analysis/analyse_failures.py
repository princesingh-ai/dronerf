import matplotlib.pyplot as plt
import seaborn as sns

from analysis.data import load_results
from analysis.utils import save_plot


sns.set_theme(style="whitegrid")

df = load_results()

false_negatives = df[
    df["prediction"] == "Non-Drone"
]

false_negatives = false_negatives.sort_values(
    "average_probability"
)

false_negatives.to_csv(
    "docs/external_val_plots/false_negatives.csv",
    index=False,
)

summary = (
    false_negatives
    .groupby("drone")
    .size()
    .reset_index(name="false_negatives")
)

summary.to_csv(
    "docs/external_val_plots/false_negative_summary.csv",
    index=False,
)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=summary,
    x="drone",
    y="false_negatives",
)

plt.title("False Negatives by Drone Model")
plt.xlabel("Drone")
plt.ylabel("False Negatives")
save_plot("false_negative_counts.png")