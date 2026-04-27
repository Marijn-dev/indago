import pandas as pd
import matplotlib.pyplot as plt

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


# --- Load CSV ---
df = pd.read_csv("pvdp/val_loss_model2.csv")
df2 = pd.read_csv("pvdp/param_loss_model2.csv")

# --- Column names ---
train_flumen = "260420_1713_flumen_gs_theta0_05 - val_loss"
train_diffrax = "260420_1713_diffrax_gs_theta0_05 - val_loss"
params_flumen = "260420_1713_flumen_gs_theta0_05 - params_loss"
params_diffrax = "260420_1713_diffrax_gs_theta0_05 - params_loss"

fig, ax = plt.subplots(figsize=(8, 5))

# Loss curves
ax.plot(
    df[train_diffrax],
    "-",
    color="#1f77b4",
    linewidth=2,
    label=r"$\ell_\theta$ (diffrax)",
)
ax.plot(
    df[train_flumen],
    "--",
    color="orange",
    linewidth=2,
    label=r"$\ell_\theta$ (flumen)",
)

# Parameter estimate curves
# ax.plot(df2[params_diffrax], "-", color="#2ca02c", linewidth=2,
#         label=r"$\hat{\theta}$ (diffrax)")
ax.plot(
    df2[params_diffrax],
    "-",
    color="#2ca02c",
    linewidth=2,
    label="RMSE (diffrax)",
)
# ax.plot(df2[params_flumen], "--", color="#d62728", linewidth=2,
#         label=r"$\hat{\theta}$ (flumen)")
ax.plot(
    df2[params_flumen],
    "--",
    color="#d62728",
    linewidth=2,
    label="RMSE (flumen)",
)

# Labels
ax.set_xlabel(r"iteration")
# ax.set_ylabel(r"$\ell_\theta  \, , \, \hat{\theta}$")
ax.set_ylabel(r"$\ell_\theta  \, $, RMSE")
# Legend
ax.legend()

plt.tight_layout()
# plt.grid(True, alpha=0.3)
plt.grid()
plt.savefig("val_loss_and_RMSE_model_v2.pdf", bbox_inches="tight")

plt.show()
