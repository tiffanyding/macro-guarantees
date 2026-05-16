
This is the code release for "Conformal prediction with macro-coverage guarantees"

# Downloading data

Our experiments make use of pre-computed softmax scores (and labels) from [Ding, Fermanian & Salmon, 2026](https://github.com/tiffanyding/long-tail-conformal), which can be downloaded and pre-processed by running

```
python download_data.py plantnet-trunc
python download_data.py inaturalist-trunc
```

The other data needed are JSON files for determining the genus groupings in `plantnet-trunc`, and this is included in the `data/` directory of this repository.  

# Reproducing experiments

To run the main experiments (targeting vanilla macro-coverage):
```
python run_main_experiments.py
```

To run the tail-focused macro-coverage experiments:
```
python run_tail_focused_experiments.py 
```

To run the genus-level macro-coverage experiments:
```
python run_genus_experiments.py 
```

Each of these scripts prints the results as a table and also saves them in Latex format to `results/`