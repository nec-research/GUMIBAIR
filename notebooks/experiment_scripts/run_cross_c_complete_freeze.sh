CONFIG_FILE="../configs/experiment_configs/cross_c_complete_freeze.yaml"
DATA_DIR=$1

time python ../train_cond_frozen_model.py $CONFIG_FILE $DATA_DIR\
    -o ../results/cross_cohort_results/cross_c_complete_freeze/ \
    --set-single