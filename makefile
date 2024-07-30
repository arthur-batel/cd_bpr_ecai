.PHONY: build, clean, check_conda

build: check_conda
	unzip data.zip
	unzip results.zip
	conda env create --file cdbpr-env.yml
clean:
	rm -rf data/
	rm -rf results/

check_conda:
	@if command -v conda >/dev/null 2>&1; then \
		echo "conda is installed"; \
	else \
		echo "conda needs to be installed\nrun the makefile again after the installation"; \
		exit 1; \
	fi