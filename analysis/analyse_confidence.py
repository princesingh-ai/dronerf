import matplotlib.pyplot as plt
import seaborn as sns

from analysis.data import load_results
from analysis.utils import save_plot


sns.set_theme(style="whitegrid")

df = load_results()

df["correct"] = df["prediction"] == "Drone"

summary = (
    df.groupby("correct")
    .agg(
        average_confidence=(
            "average_probability",
            "mean",
        ),
        recordings=(
            "prediction",
            "count",
        ),
    )
    .reset_index()
)

summary.to_csv(
    "docs/external_val_plots/confidence_summary.csv",
    index=False,
)

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="average_probability",
    bins=25,
    hue="correct",
    multiple="stack",
)

plt.title("Prediction Confidence Distribution")
plt.xlabel("Average Probability")
plt.ylabel("Recordings")

save_plot("confidence_histogram.png")