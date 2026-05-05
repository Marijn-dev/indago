from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

# --- Matplotlib settings ---
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 12,
    }
)


def parse_args():

    ap = ArgumentParser()

    ap.add_argument(
        "results_one", type=str, help="Path to pkl estimation results"
    )

    ap.add_argument(
        "results_two", type=str, help="Path to pkl estimation results"
    )

    return ap.parse_args()


def main():
    args = parse_args()

    # Load in data
    data_path = Path(args.results_one)
    with data_path.open("rb") as f:
        data_one = pickle.load(f)
    data_path = Path(args.results_two)
    with data_path.open("rb") as f:
        data_two = pickle.load(f)
    est_params_one = np.asarray(data_one["est_params"])
    est_params_two = np.asarray(data_two["est_params"])
    val_losses_one = data_one["val_losses"]
    val_losses_two = data_two["val_losses"]
    method_one = data_one["method"]
    method_two = data_two["method"]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Loss plot
    ax.plot(
        val_losses_one,
        "-",
        color="#1f77b4",
        linewidth=2,
        label=rf"$\ell_\theta$ ({method_one})",
    )
    ax.plot(
        val_losses_two,
        "--",
        color="orange",
        linewidth=2,
        label=rf"$\ell_\theta$ ({method_two})",
    )

    # Est parameter plot
    n_params = est_params_one.shape[1]
    # Color palettes (one per method)
    colors_one = ["#2ca02c", "#98df8a", "#1b7f1b", "#66c266"]  # greens
    colors_two = ["#d62728", "#ff9896", "#a50f15", "#e6550d"]  # reds

    for i in range(n_params):
        # Method one (green shades)
        ax.plot(
            est_params_one[:, i],
            "-",
            color=colors_one[i % len(colors_one)],
            linewidth=2,
            label=rf"$\hat{{\theta}}_{i + 1}$ ({method_one})",
        )

        # Method two (red shades)
        ax.plot(
            est_params_two[:, i],
            "--",
            color=colors_two[i % len(colors_two)],
            linewidth=2,
            label=rf"$\hat{{\theta}}_{i + 1}$ ({method_two})",
        )

    ax.set_xlabel(r"Step")
    ax.set_ylabel(r"$\ell_\theta  \, , \, \hat{\theta}$")
    ax.legend()

    plt.tight_layout()
    plt.grid()
    save_dir = os.path.dirname(os.path.dirname(args.results_one))
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/results.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
