.PHONY: env_run_experiments
env_run_experiments:
	conda env create -f env_run_experiments.yml && \
	conda run -n env_run_experiments pip install . && \
	cd experiments && \
	conda run -n env_run_experiments pip install .
