import matplotlib.pyplot as plt
import seaborn as sns

from analysis.data import load_results
from analysis.utils import save_plot


sns.set_theme(style="whitegrid")

df = load_results()

summary = (
    df.groupby("interference")
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

summary.to_csv("docs/external_val_plots/benchmark_summary.csv", index=False)

plt.figure(figsize=(7,5))
sns.barplot(data=summary, x="interference", y="accuracy")

plt.ylim(0,1)

plt.title("External Benchmark Accuracy")
plt.xlabel("")
plt.ylabel("Accuracy")
save_plot("overall_accuracy.png")