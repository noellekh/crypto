# $DEL_BEGIN

####### 👇 SIMPLE SOLUTION (x86 and M1) 👇 ########
FROM python:3.10.6-slim-buster
COPY scripts/jh_crypto-run scripts/jh_crypto-run
COPY jh_crypto jh_crypto
COPY cryptogcloud-4279ebbabaea.json cryptogcloud-4279ebbabaea.json
COPY requirements.txt requirements.txt
COPY setup.py setup.py
RUN pip install .

COPY .env .env

RUN python -c 'from dotenv import load_dotenv, find_dotenv; load_dotenv(find_dotenv()); \
    from jh_crypto.ml_logic.registry import load_model; load_model()'

CMD uvicorn jh_crypto.api.fast:app --host 0.0.0.0 --port $PORT

# ####### 👇 OPTIMIZED SOLUTION (x86)👇  (May be too advanced for ML-Ops module but useful for the project weeks) #######

# # tensorflow base-images are optimized: lighter than python-buster + pip install tensorflow
# FROM tensorflow/tensorflow:2.10.0
# # OR for apple silicon, use this base image instead
# # FROM armswdev/tensorflow-arm-neoverse:r22.09-tf-2.10.0-eigen

# # WORKDIR /prod
# COPY jh_crypto /jh_crypto

# # We strip the requirements from useless packages like `ipykernel`, `matplotlib` etc...
# COPY requirements.txt /requirements.txt
# COPY setup.py setup.py
# RUN pip install .

# # Copy .env with DATA_SOURCE=local and MODEL_TARGET=mlflow
# COPY .env .env

# # A build time, download the model from the MLflow server and copy it once for all inside of the image
# RUN python -c 'from dotenv import load_dotenv, find_dotenv; load_dotenv(find_dotenv()); \
#     from jh_crypto.ml_logic.registry import load_model; load_model()'

# # Then, at run time, load the model locally from the container instead of querying the MLflow server, thanks to "MODEL_TARGET=local"
# # This avoids to download the heavy model from the Internet every time an API request is performed
# CMD MODEL_TARGET=local uvicorn jh_crypto.api.fast:app --host 0.0.0.0 --port $PORT

# # $DEL_END
