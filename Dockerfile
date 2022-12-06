# $DEL_BEGIN

####### 👇 SIMPLE SOLUTION (x86 and M1) 👇 ########

FROM python:3.10.6-slim-buster
COPY scripts/crypto-run scripts/crypto-run
COPY crypto crypto
COPY cryptogcloud-4279ebbabaea.json cryptogcloud-4279ebbabaea.json
COPY requirements_prod.txt requirements.txt
COPY setup.py setup.py
RUN pip install .

COPY .env .env

RUN python -c 'from dotenv import load_dotenv, find_dotenv; load_dotenv(find_dotenv()); \
    from crypto.ml_logic.registry import load_model; load_model()'

CMD uvicorn crypto.api.fast:app --host 0.0.0.0 --port $PORT

# ####### 👇 OPTIMIZED SOLUTION (x86)👇  (May be too advanced for ML-Ops module but useful for the project weeks) #######

# FROM tensorflow/tensorflow:latest

# COPY scripts/crypto-run scripts/crypto-run
# COPY crypto crypto
# COPY cryptogcloud-4279ebbabaea.json cryptogcloud-4279ebbabaea.json

# COPY requirements_prod.txt /requirements.txt
# COPY setup.py setup.py
# RUN pip install .
# COPY .env .env

# RUN python -c 'from dotenv import load_dotenv, find_dotenv; load_dotenv(find_dotenv()); \
#     from crypto.ml_logic.registry import load_model; load_model()'

# CMD MODEL_TARGET=local uvicorn crypto.api.fast:app --host 0.0.0.0 --port $PORT

# $DEL_END
