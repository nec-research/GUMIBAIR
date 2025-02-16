data_modality?="abundance"
skiprows=209

.PHONY: env_run_experiments
env_run_experiments:
	conda env create -f env_run_experiments.yml && \
	conda run -n env_run_experiments pip install . && \
	cd experiments && \
	conda run -n env_run_experiments pip install .

.PHONY: env_fetch_and_curate_data
env_fetch_and_curate_data:
	conda env create -f env_fetch_and_curate_data.yml

.PHONY: fetch_data
fetch_data:
	if [ ! -d "data_pipeline/data" ]; then \
		mkdir data_pipeline/data; \
	fi
	cd data_pipeline && \
	conda run -n env_fetch_and_curate_data Rscript ./fetch/get_data_from_cMD.R

.PHONY: curate_data
curate_data:
	if [ ! -d "data_pipeline/data/normalized" ]; then \
		mkdir data_pipeline/data/normalized; \
	fi

	cd data_pipeline && \
	conda run -n env_fetch_and_curate_data python ./curate/normalize_data.py ./data/ -d marker -s $(skiprows) && \
	conda run -n env_fetch_and_curate_data python ./curate/run_checks.py ./data/normalized/ -d marker -sk $(skiprows)

.PHONY: data
data: env_fetch_and_curate_data fetch_data curate_data