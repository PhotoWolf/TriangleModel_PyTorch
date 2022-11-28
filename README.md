# TriangleModel_PyTorch
A lightweight PyTorch implementation of the Triangle Model [(Harm and Seidenberg, 2004)](https://psycnet.apa.org/record/2004-15929-005). 
- *model.py*: Defines the gradients of the system.
- *train.py*: Training utilities and ODE solver.
- *benchmark.py*: Runs phased training procedure. Saves losses / accuracies.
- *dataset.py*: Data utilities and tokenizer class.
- *evaluation.py*: Runs evaluation pipeline. Plots results.

To run the default set of experiments,
```
python3 benchmaking.py -ID baseline -model_config configs/baseline/model_config.json -trainer_config configs/baseline/trainer_config.json -optimizer_config configs/baseline/opt_config.json
python3 evaluation.py -ID baseline -model_config configs/baseline/model_config.json -trainer_config configs/baseline/trainer_config.json
```
