CONFIGS_LOCATION="../configs/experiment_configs/in_c_ind_cohorts/"
DATA_DIR=$1

for config_fname in $CONFIGS_LOCATION/*; do
    time python ../run_experiments.py in_cohort $config_fname $DATA_DIR \
        --replicates 5 \
        --output-path "./" \
        --benchmark \
        --cross-validate
done