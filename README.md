# indago
A simple package for estimating parameters in control systems.

First, create data using of some parameterised system with the Delta distribution. For example, generate data of the Fitzhugh-Nagumo model using the following command:
```shell
  python scripts/semble_generate.py --n_trajectories 100 --n_samples 200 data_generation/fhn.yaml fhn_test_data
```
This will create a data file in `./data/fhn_test_data.pkl`.

Then we can use this data and a model trained using the [`flumen-jax`](https://github.com/Marijn-dev/flumen-jax/tree/parameterised-dynamics) package to estimate the parameters of the dynamics using the following command:
```shell
  python scripts/estimate_wandb.py data/fhn_test_data.pkl <model_path> fhn_test
```
where model ```model_path``` should be replaced by a Weights and Biases model.

