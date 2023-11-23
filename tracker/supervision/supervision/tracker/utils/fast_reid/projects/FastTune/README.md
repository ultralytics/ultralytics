# Hyper-Parameter Optimization in FastReID

This project includes training reid models with hyper-parameter optimization.

Install the following

```bash
pip install 'ray[tune]'
pip install hpbandster ConfigSpace hyperopt
```

## Example

This is an example for tuning `batch_size` and `num_instance` automatically.

To train hyperparameter optimization with BOHB(Bayesian Optimization with HyperBand) search algorithm, run

```bash
python3 projects/FastTune/tune_net.py --config-file projects/FastTune/configs/search_trial.yml --srch-algo "bohb"
```

## Known issues
todo