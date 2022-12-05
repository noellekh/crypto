# $CODE_BEGIN
from crypto.interface.main import preprocess, train, evaluate

from prefect import task, Flow, Parameter

import os
import requests
# $CODE_END

@task
def preprocess_new_data(experiment):
    """
    Run the preprocessing of the new data
    """
    # $CHA_BEGIN
    preprocess()
    preprocess(source_type='val')
    # $CHA_END

@task
def evaluate_production_model(status):
    """
    Run the `Production` stage evaluation on new data
    Returns `eval_mae`
    """
    # $CHA_BEGIN
    eval_mae = evaluate()
    return eval_mae
    # $CHA_END

@task
def re_train(status):
    """
    Run the training
    Returns train_mae
    """
    # $CHA_BEGIN
    train_mae = train()
    return train_mae
    # $CHA_END

# $WIPE_BEGIN
@task
def notify(eval_mae, train_mae):
    base_url = 'https://wagon-chat.herokuapp.com'
    channel = 'krokrob'
    url = f"{base_url}/{channel}/messages"
    author = 'krokrob'
    content = "Evaluation MAE: {} - New training MAE: {}".format(
        round(eval_mae, 2), round(train_mae, 2))
    data = dict(author=author, content=content)
    response = requests.post(url, data=data)
    response.raise_for_status()
# $WIPE_END

def build_flow():
    """
    build the prefect workflow for the `taxifare` package
    """
    # $CHA_BEGIN
    flow_name = os.environ.get("PREFECT_FLOW_NAME")

    with Flow(flow_name) as flow:

        # retrieve mlfow env params
        mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT")

        # create workflow parameters
        experiment = Parameter(name="experiment", default=mlflow_experiment)

        # register tasks in the workflow
        status = preprocess_new_data(experiment)
        eval_mae = evaluate_production_model(status)
        train_mae = re_train(status)
        notify(eval_mae, train_mae)

    return flow
    # $CHA_END
