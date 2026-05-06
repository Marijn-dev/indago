# indago
A package for estimating parameters in control systems. 

This repository also contains the data and scripts used to reproduce and generate the results presented in [Miguel Aguiar, Marijn Ruiter, Amritam Das and Karl H. Johansson, _Flumen: Flow function learning for surrogate modelling of control systems (2026)]

The corresponding models and data used in the paper can be found in the `./models` and `./data` folders, respectively.

The models are trained using the [`flumen-jax`](https://github.com/Marijn-dev/flumen-jax/tree/parameterised-dynamics) package.

The data for a parameter estimation experiment can be created using the `./scripts/create_data.py` script. For example, to generate the Van der Pol data used in Table 1 (i):
```shell
  python scripts/creat_data.py --n_trajectories 100 --n_samples 200 --time_horizon 15 data/vdp/vdp.yaml vdp_M60
```
This will create a data file in `./data/vdp/vdp_M60.pkl`.

