# indago

A package for estimating parameters in control systems. 

This repository also contains the data and scripts used to reproduce and generate the results presented in [Miguel Aguiar, Marijn Ruiter, Amritam Das and Karl H. Johansson, _Flumen: Flow function learning for surrogate modelling of control systems (2026)]

The Flumen models are trained using the [`flumen-jax`](https://github.com/Marijn-dev/flumen-jax/tree/parameterised-dynamics) package. The trained models used in the paper can be found in the `./models` folder.

To create the data, the [`semble`](https://github.com/Marijn-dev/semble/tree/parameterised-dynamics) package is required. The data used for the parameter estimation experiments in the paper can be found in the `./data` folder. The data used for training (and calculation of the test set RRMSE error) can be downloaded [HERE].

## Test Set Performance and Trajectory Visualization
The reported RRMSE test losses in the paper can be calculated using the `./scripts/create_data.py` script:
```shell
  python scripts/test_loss.py [data_path] [model_path]
```
where `[data_path]` and `[model_path]` are the paths to the dataset and model, respectively.

To simulate and visualize some model predictions, use the `./scripts/trajectory_simulation_vdp.py` and `./scripts/trajectory_simulation_ctm.py` scripts:
```shell
  python scripts/trajectory_simulation_vdp.py models/vdp/
```
and 
```shell
  python scripts/trajectory_simulation_ctm.py models/ctm/
```
which will save the plots in figures 3 and 5 in the folders `./results/trajectory_simulation/vdp/` and `./results/trajectory_simulation/ctm/`, respectively. To simulate different trajectories, change the NUMPY_RNG_SEED value in the script.

## Simulation Performance
To create the trajectory simulation result in the figure 6 (a), use the `./scripts/time_traj_eval.py` script:
```shell
  python scripts/time_traj_eval.py --n_traj_samples 50 --n_time_samples 100 --time_horizons 25 --dts 0.002 0.001 0.0005
```
which will save the figure in `./results/simulation_performance/ctm/traj_timings.pdf`

To create the gradient computation result in the figure 6 (b), use the `./scripts/time_grad_eval.py` script:
```shell
  python scripts/time_grad_eval.py --n_traj_samples 50 --time_horizons 25 --dts 0.002 0.001 0.0005 0.0002
```
which will save the figure in `./results/simulation_performance/ctm/grad_timings.pdf`

## Parameter Estimation
The data for a parameter estimation experiment can be created using the `./scripts/semble_generate.py` script. For example, to generate Van der Pol data with the same settings as in Table 1 (i):
```shell
  python scripts/semble_generate.py --n_trajectories 100 --n_samples 200 --time_horizon 15 data_generation/vdp.yaml vdp_M_60
```
This will create a data file in `./data/vdp_M_60.pkl`.

### Single Experiment
This data can be used for a parameter estimation experiment, using either `./scripts/estimate_local.py` or `./scripts/estimate_wandb.py`. For example, to recreate the result in Table 1 (i) that uses gradient descent and $\hat{\theta}_0=0.5$:
```shell
  python scripts/estimate_local.py data/vdp_M_60.pkl models/vdp/ GradientDescent 0.5
```
and 
```shell
  python scripts/estimate_local.py data/vdp_M_60.pkl Tsit5 GradientDescent 0.5 --dt 0.01
```
which will also save the results in `./results/estimation/vdp/Flumen/results_dict.pkl` and `./results/estimation/vdp/Tsit5/results_dict.pkl`, respectively.

These results can then be analyzed and compared using the `./scripts/estimate_eval.py` script. To recreate figure 4:
```shell
  python scripts/estimate_eval.py results/estimation/vdp/Flumen/results_dict.pkl results/estimation/vdp/Tsit5/results_dict.pkl
```
this will save the corresponding figure in `./results/estimation/vdp/results.pdf`.

### Monte Carlo Experiment
To run a Monte carlo experiment, use either `./scripts/MC_local.py` or `./scripts/MC_wandb.py`. For example, to recreate the results used in figures 7 and 8:
```shell
  python scripts/MC_local.py data/ctm_M_60.pkl models/ctm/ BFGS parameter_generation/ctm.yaml --n_runs 50 
```
and
```shell
  python scripts/MC_local.py data/ctm_M_60.pkl Euler BFGS parameter_generation/ctm.yaml --n_runs 50 --dt 0.002
```
which will create results in `./results/MC/ctm/Flumen/results.dict.pkl` and `./results/MC/ctm/Euler/results.dict.pkl`, respectively.

Then, these results can be analyzed using `./scripts/MC_eval.py`. To recreate figures 7 and 8:
```shell
  python scripts/MC_eval.py data/ctm_M_60.pkl results/MC/ctm/Flumen/results_dict.pkl results/MC/ctm/Euler/results_dict.pkl
```
which will save the figures in the `./results/MC/ctm/` folder.