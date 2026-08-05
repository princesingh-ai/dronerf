import matplotlib.pyplot as plt
import seaborn as sns

from analysis.data import load_results
from analysis.utils import save_plot


sns.set_theme(style="whitegrid")

df = load_results()

summary = (
    df.groupby("flight_mode")
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
    "flight_mode",
)

summary.to_csv(
    "docs/external_val_plots/flight_mode_accuracy.csv",
    index=False,
)

plt.figure(figsize=(6, 5))

sns.barplot(
    data=summary,
    x="flight_mode",
    y="accuracy",
)

plt.ylim(0, 1)
plt.title("Accuracy by Flight Mode")
plt.xlabel("Flight Mode")
plt.ylabel("Accuracy")
save_plot("flight_mode_accuracy.png")