data="data/"
data_normalized="data/normalized/"
data_modality?="abundance"
skiprows=209

run_pipeline:
	Rscript fetch/get_data_from_cMD.R
	python curate/normalize_data.py $(data) -d marker -s $(skiprows)
	python curate/normalize_data.py $(data) -d abundance -s $(skiprows)
	python curate/run_checks.py $(data_normalized) -d $(data_modality) -s $(skiprows)

fetch_data: 
	Rscript get_data_from_cMD.R

normalize_data:
	python curate/normalize_data.py $(data) -d $(data_modality) -s $(skiprows)

run_checks:
	python curate/run_checks.py $(data_normalized) -d $(data_modality) -s $(skiprows)


