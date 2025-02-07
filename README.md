## GUMIBAIR

### Installation
```gumibair``` can be installed as a Python package **(Python 3.6, PyTorch 1.7, CUDA 10.2)**.
```
cd gumibair
pip install .
```

### Experiments
#### Implementation
A package with a simple API to run a bunch of experiments using `gumibair` can be found in the `experiments/mcbn_experiments/` directory and can be installed with:
```
cd experiments
pip install .
```

The package consists of a couple of different modules
##### utils.py
Implements functions to parse `.yaml` configurations for training as well as to train GUMIBAIR and RF. Additionally, it includes two custom functions to split the dataset for cross-cohort experiments (holding out one cohort either partially or complete).

##### base_exp.py, in_cohort_exp.py $ cross_cohort_exp.py
The two classes to be used are namely `InCohortExperiment` and `CrossCohortExperiment`, which are both based on a base class `_BaseExperiment`.\
An instance of either of those two classes contains a reference to a `FullMicrobiomeDataset` object as well as a config (containing hyperparameters etc.) and a number of replicates.\
For both classes, the experiment can be benchmarked against *MVIB* and *RF* directly by setting `benchmark=True` during instantiating of the experiment object.\
A method called `set_ids()` is implemented specific to which class the instance belongs to, based on the splitting functions from `utils.py`. When the experiment is run, `set_ids()` defines the train/val/test ids and labels for a specific random seed. All replicates from an instance of one of the classes can be executed with: 
```
Instance.run_replicates()
```
For the `InCohortExperiment` class, the `run_replicates()` method returns a tuple of format `(condensed_scores, per_cohort_scores)`, where each value in the tuple is a list with `pd.DataFrame` objects and each object in a list represents the scores from one replicate with a particular random seed.\
For the `CrossCohortExperiment` class, the `run_replicates()` method returns only one list `condensed_scores`, which contains `pd.DataFrame` objects with the condensed scores from all replicates, each dataframe again representing one replicate with one particular random seed.

#### Running an Experiment
The experiment classes can either be used interactively in a `.ipynb` file or via the script `run_experiment.py`:
```
python run_experiments.py {in_cohort,cross_cohort} {config_file} 
    --output-path (default: './')
    --test-prop (default: 0.9)
    --test-on-best-condition
    --mode ({partial,complete}, default: 'partial')
    --replicates (default: 5)
    --benchmark
    --cross_validate
```
The `test-on-best-condition` flag is only to be used in combination with the `partial` mode. When set, the cohort index from all cohorts execpt the heldout cohort is used for conditioning on the sampels from the heldout cohort that are found in the validation set of the current replicate. The best-working condition (based on validation ROC AUC) is then used during training. If no samples of the heldout cohort are found in the validation set, the step is skipped and the cohort index remains unchanged for that replicate and cohort.\
<br/>
When setting the `--cross_validate` flag (can only be used with `in_cohort` option), each replicate runs in a 5-fold cross validation setup, where the dataset is split into train and test 5 times. The predictions from all 5 folds are concatenated to compute the overall performance scores for the replicate.\