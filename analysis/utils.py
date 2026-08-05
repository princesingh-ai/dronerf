from pathlib import Path

import matplotlib.pyplot as plt


PLOTS_DIR = Path("docs/external_val_plots")

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def save_plot(filename: str) -> None:
    """Save the current plot to the plots directory."""
    plt.tight_layout()

    plt.savefig(PLOTS_DIR / filename, dpi=300)

    plt.close()