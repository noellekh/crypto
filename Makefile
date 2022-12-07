# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
reinstall_package:
	@pip uninstall -y crypto || :
	@pip install -e .

install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* crypto/*.py

black:
	@black scripts/* crypto/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr crypto-*.dist-info
	@rm -fr crypto.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)



run_api:
	uvicorn crypto.api.fast:app --reload


root:
	@printf "\n" && curl -X 'GET' \
  'http://localhost:8000/' \
  -H 'accept: application/json' && printf "\n"


push_datasets:
# -bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.BTC-USDT_processed_1j ${LOCAL_DATA_PATH}/processed/BTC-USDT_processed_1j.csv
# -bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.UNI-USDT_processed_1j ${LOCAL_DATA_PATH}/processed/UNI-USDT_processed_1j.csv
# -bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.MATIC-USDT_processed_1j ${LOCAL_DATA_PATH}/processed/MATIC-USDT_processed_1j.csv
# -bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.DOGE-USDT_processed_1j ${LOCAL_DATA_PATH}/processed/DOGE-USDT_processed_1j.csv
# -bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.ATOM-USDT_processed_1j ${LOCAL_DATA_PATH}/processed/ATOM-USDT_processed_1j.csv
# -bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.ETH-USDT_processed_1j ${LOCAL_DATA_PATH}/processed/ETH-USDT_processed_1j.csv
# -bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.BNB-USDT_processed_1j ${LOCAL_DATA_PATH}/processed/BNB-USDT_processed_1j.csv
# -bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.ADA-USDT_processed_1j ${LOCAL_DATA_PATH}/processed/ADA-USDT_processed_1j.csv
# -bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.LTC-USDT_processed_1j ${LOCAL_DATA_PATH}/processed/LTC-USDT_processed_1j.csv
# -bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.BTC-USDT ${LOCAL_DATA_PATH}/raw/BTC-USDT.csv
# -bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.UNI-USDT ${LOCAL_DATA_PATH}/raw/UNI-USDT.csv
# -bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.MATIC-USDT ${LOCAL_DATA_PATH}/raw/MATIC-USDT.csv
# -bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.DOGE-USDT ${LOCAL_DATA_PATH}/raw/DOGE-USDT.csv
# -bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.ATOM-USDT ${LOCAL_DATA_PATH}/raw/ATOM-USDT.csv
# -bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.ETH-USDT ${LOCAL_DATA_PATH}/raw/ETH-USDT.csv
# -bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.BNB-USDT ${LOCAL_DATA_PATH}/raw/BNB-USDT.csv
# -bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.ADA-USDT ${LOCAL_DATA_PATH}/raw/ADA-USDT.csv
# -bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.LTC-USDT ${LOCAL_DATA_PATH}/raw/LTC-USDT.csv

	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.BTC-USDT_processed_1d ${LOCAL_DATA_PATH}/processed/BTC-USDT_processed_1d.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.UNI-USDT_processed_1d ${LOCAL_DATA_PATH}/processed/UNI-USDT_processed_1d.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.MATIC-USDT_processed_1d ${LOCAL_DATA_PATH}/processed/MATIC-USDT_processed_1d.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.DOGE-USDT_processed_1d ${LOCAL_DATA_PATH}/processed/DOGE-USDT_processed_1d.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.ATOM-USDT_processed_1d ${LOCAL_DATA_PATH}/processed/ATOM-USDT_processed_1d.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.ETH-USDT_processed_1d ${LOCAL_DATA_PATH}/processed/ETH-USDT_processed_1d.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.BNB-USDT_processed_1d ${LOCAL_DATA_PATH}/processed/BNB-USDT_processed_1d.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.ADA-USDT_processed_1d ${LOCAL_DATA_PATH}/processed/ADA-USDT_processed_1d.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.LTC-USDT_processed_1d ${LOCAL_DATA_PATH}/processed/LTC-USDT_processed_1d.csv


	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.BTC-USDT_processed_12h ${LOCAL_DATA_PATH}/processed/BTC-USDT_processed_12h.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.UNI-USDT_processed_12h ${LOCAL_DATA_PATH}/processed/UNI-USDT_processed_12h.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.MATIC-USDT_processed_12h ${LOCAL_DATA_PATH}/processed/MATIC-USDT_processed_12h.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.DOGE-USDT_processed_12h ${LOCAL_DATA_PATH}/processed/DOGE-USDT_processed_12h.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.ATOM-USDT_processed_12h ${LOCAL_DATA_PATH}/processed/ATOM-USDT_processed_12h.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.ETH-USDT_processed_12h ${LOCAL_DATA_PATH}/processed/ETH-USDT_processed_12h.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.BNB-USDT_processed_12h ${LOCAL_DATA_PATH}/processed/BNB-USDT_processed_12h.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.ADA-USDT_processed_12h ${LOCAL_DATA_PATH}/processed/ADA-USDT_processed_12h.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.LTC-USDT_processed_12h ${LOCAL_DATA_PATH}/processed/LTC-USDT_processed_12h.csv

	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.BTC-USDT_processed_6h ${LOCAL_DATA_PATH}/processed/BTC-USDT_processed_6h.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.UNI-USDT_processed_6h ${LOCAL_DATA_PATH}/processed/UNI-USDT_processed_6h.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.MATIC-USDT_processed_6h ${LOCAL_DATA_PATH}/processed/MATIC-USDT_processed_6h.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.DOGE-USDT_processed_6h ${LOCAL_DATA_PATH}/processed/DOGE-USDT_processed_6h.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.ATOM-USDT_processed_6h ${LOCAL_DATA_PATH}/processed/ATOM-USDT_processed_6h.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.ETH-USDT_processed_6h ${LOCAL_DATA_PATH}/processed/ETH-USDT_processed_6h.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.BNB-USDT_processed_6h ${LOCAL_DATA_PATH}/processed/BNB-USDT_processed_6h.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.ADA-USDT_processed_6h ${LOCAL_DATA_PATH}/processed/ADA-USDT_processed_6h.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.LTC-USDT_processed_6h ${LOCAL_DATA_PATH}/processed/LTC-USDT_processed_6h.csv
