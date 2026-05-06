# indago
A package for estimating parameters in control systems. 

This repository also contains the data and scripts used to reproduce and generate the results presented in [Miguel Aguiar, Marijn Ruiter, Amritam Das and Karl H. Johansson, _Flumen: Flow function learning for surrogate modelling of control systems (2026)]

The corresponding models and data used in the paper can be found in the `./models` and `./data` folders, respectively.

The Flumen models are trained using the [`flumen-jax`](https://github.com/Marijn-dev/flumen-jax/tree/parameterised-dynamics) package.

The data for a parameter estimation experiment can be created using the `./scripts/create_data.py` script. For example, to generate Van der Pol data with the same settings as in Table 1 (i):
```shell
  python scripts/create_data.py --n_trajectories 100 --n_samples 200 --time_horizon 15 data/vdp/vdp.yaml M_60
```
This will create a data file in `./data/vdp/M_60.pkl`.

Then, this data can be used for a parameter estimation experiment, using either `./scripts/estimate_local.py` or `./scripts/estimate_wandb.py`. For example, to recreate a result in Table 1 (i) using gradient descent and $\hat{\theta}_0=0.5$:
```shell
  python scripts/estimate_local.py data/vdp/M_60.pkl Flumen GradientDescent 0.5
```
and 
```shell
  python scripts/estimate_local.py data/vdp/M_60.pkl Tsit5 GradientDescent 0.5 --dt 0.01
```
which will also save the results in `./results/estimation/vdp/Flumen/results_dict.pkl` and `./results/estimation/vdp/Tsit5/results_dict.pkl`, respectively.

These results can then be analyzed and compared using the `./scripts/estimate_eval.py` script. For example, to recreate figure 4:
```shell
  python scripts/estimate_eval.py results/estimation/vdp/Flumen/results_dict.pkl results/estimation/vdp/Tsit5/results_dict.pkl
```
this will save the corresponding figure in `./results/estimation/vdp/results.pdf`.

To run a Monte carlo experiment, use either `./scripts/MC_local.py` or `./scripts/MC_wandb.py`. For example, to recreate the results used to generate figure 6 and 7:
```shell
  python scripts/MC_local.py data/ctm/M_60.pkl Flumen BFGS parameter_generation/ctm_param.yaml --n_runs 50 
```
and
```shell
  python scripts/MC_local.py data/ctm/M_60.pkl Euler BFGS parameter_generation/ctm_param.yaml --n_runs 50 --dt 0.002
```
which will create results in `./results/MC/ctm/Flumen/results.dict.pkl` and `./results/MC/ctm/Euler/results.dict.pkl`, respectively.

Then, these results can be analyzed using `./scripts/MC_eval.py`. For example, to recreate figures 6 and 7:
```shell
  python scripts/MC_eval.py results/MC/ctm/Flumen/results_dict.pkl results/estimation/vdp/Tsit5/results_dict.pkl
```
which will save the figures in the `./results/MC/ctm/` folder.