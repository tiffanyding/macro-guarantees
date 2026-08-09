
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
python run_main_experiments.py plantnet-trunc
python run_main_experiments.py inaturalist-trunc
```

To run the tail-focused macro-coverage experiments:
```
python run_tail_focused_experiments.py 
```

To run the genus-level macro-coverage experiments:
```
python run_genus_experiments.py 
```

Additional baseline methods: to rerun the main experiments with the additional baseline methods Clustered CP (Ding et al., 2023) and RC3P (Shi et al., 2024), rerun `run_main_experiments.py` with the flag `--methods "clustered, rc3p"`

Each of these scripts prints the results as a table and also saves them in Latex format to `results/`