import matplotlib.pyplot as plt
import seaborn as sns

from analysis.data import load_results
from analysis.utils import save_plot


sns.set_theme(style="whitegrid")

df = load_results()

summary = (
    df.groupby("drone")
    .agg(
        accuracy=(
            "prediction",
            lambda x: (x == "Drone").mean(),
        ),
        recordings=(
            "prediction",
            "count",
        ),
    )
    .reset_index()
)

summary = summary.sort_values(
    "accuracy",
    ascending=False,
)

summary.to_csv(
    "docs/external_val_plots/drone_accuracy.csv",
    index=False,
)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=summary,
    x="drone",
    y="accuracy",
)

plt.ylim(0, 1)
plt.title("Accuracy by Drone Model")
plt.xlabel("Drone")
plt.ylabel("Accuracy")
save_plot("drone_accuracy.png")