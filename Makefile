DOWNLOAD_PATH ?= ./data

test:
	python -m pytest --cov-config .coveragerc --cov=./dosed --cov-fail-under=90 --cov-report=term-missing

download_example:
	@echo Dowloading Data and converting to H5 in $(DOWNLOAD_PATH)
	python minimum_example/download_data.py $(DOWNLOAD_PATH)/downloads/
	python minimum_example/to_h5.py $(DOWNLOAD_PATH)/downloads/ $(DOWNLOAD_PATH)/h5/

start_docker:
	docker build -t lol .
	docker run --runtime=nvidia -it -v "${PWD}:/workspace" lol bash
