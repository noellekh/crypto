from crypto.ml_logic.params import LOCAL_REGISTRY_PATH

import mlflow
from mlflow.tracking import MlflowClient

import glob
import os
import time
import pickle

from colorama import Fore, Style

from tensorflow.keras import Model, models


def save_model(model: Model = None,
               params: dict = None,
               metrics: dict = None,
               pair: str = "BTC-USDT",
               freq: str = "1d") -> None:
    """
    persist trained model, params and metrics
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    if os.environ.get("MODEL_TARGET") == "mlflow":

        # retrieve mlflow env params
        # $CHA_BEGIN
        mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT")
        mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")+pair+"_"+freq
        # $CHA_END

        # configure mlflow
        # $CHA_BEGIN
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name=mlflow_experiment)
        # $CHA_END

        with mlflow.start_run():

            # STEP 1: push parameters to mlflow
            # $CHA_BEGIN
            if params is not None:
                mlflow.log_params(params)
            # $CHA_END

            # STEP 2: push metrics to mlflow
            # $CHA_BEGIN
            if metrics is not None:
                mlflow.log_metrics(metrics)
            # $CHA_END

            # STEP 3: push model to mlflow
            # $CHA_BEGIN
            if model is not None:

                mlflow.keras.log_model(keras_model=model,
                                       artifact_path="model",
                                       keras_module="tensorflow.keras",
                                       registered_model_name=mlflow_model_name)
            # $CHA_END

        # $WIPE_BEGIN
        print("\n✅ data saved to mlflow")
        # $WIPE_END

        return None

    print(Fore.BLUE + "\nSave model to local disk..." + Style.RESET_ALL)

    # save params
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
        print(f"- params path: {params_path}")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # save metrics
    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")
        print(f"- metrics path: {metrics_path}")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    # save model
    if model is not None:
        model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", timestamp)
        print(f"- model path: {model_path}")
        model.save(model_path)

    print("\n✅ data saved locally")

    return None


def load_model(save_copy_locally=False,pair: str="BTC-USDT",freq: str= "1d") -> Model:
    """
    load the latest saved model, return None if no model found
    """
    if os.environ.get("MODEL_TARGET") == "mlflow":
        stage = get_model_version(pair=pair,freq=freq)
        # print(stage)

        print(Fore.BLUE + f"\nLoad model stage from mlflow..." + Style.RESET_ALL)

        # load model from mlflow
        model = None
        # $CHA_BEGIN
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

        mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")+pair+"_"+freq

        model_uri = f"models:/{mlflow_model_name}/{stage}"
        print(f"- uri: {model_uri}")

        try:
            model = mlflow.keras.load_model(model_uri=model_uri)
            print("\n✅ model loaded from mlflow")
        except:
            print(f"\n❌ no model in stage on mlflow")
            return None

        if save_copy_locally:
            from pathlib import Path

            # Create the LOCAL_REGISTRY_PATH directory if it does exist
            Path(LOCAL_REGISTRY_PATH).mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", timestamp)
            model.save(model_path)
        # $CHA_END

        return model

    print(Fore.BLUE + "\nLoad model from local disk..." + Style.RESET_ALL)

    # get latest model version
    model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")

    results = glob.glob(f"{model_directory}/*")
    if not results:
        return None

    model_path = sorted(results)[-1]
    print(f"- path: {model_path}")

    model = models.load_model(model_path)
    print("\n✅ model loaded from disk")

    return model


def get_model_version(stage="Production",pair:str="BTC-USDT",freq:str="1d"):
    """
    Retrieve the version number of the latest model in the given stage
    - stages: "None", "Production", "Staging", "Archived"
    """

    if os.environ.get("MODEL_TARGET") == "mlflow":

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

        mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")+pair+"_"+freq

        client = MlflowClient()

        try:
            # version = client.get_latest_versions(name=mlflow_model_name, stages=[stage])
            version = client.get_latest_versions(name=mlflow_model_name)
        except:
            return None

        # check whether a version of the model exists in the given stage
        if not version:
            return None

        return int(version[0].version)

    # model version not handled

    return None
