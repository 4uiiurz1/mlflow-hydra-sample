import mlflow
from mlflow.tracking.client import MlflowClient
from omegaconf import DictConfig, ListConfig


def start_run(run_id=None, experiment_id=None, run_name=None, nested=False):
    experiment_ids = [
        exp.experiment_id for exp in MlflowClient().list_experiments()]

    if run_id is None and run_name is not None:
        runs = mlflow.search_runs(
            experiment_ids=experiment_ids if experiment_id is None else [experiment_id])
        runs.head()
    mlflow.start_run()


def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        if isinstance(element, DictConfig):
            for k, v in element.items():
                if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                    explore_recursive(f'{param_name}.{k}', v)
                else:
                    mlflow.log_param(f'{param_name}.{k}', v)
        elif isinstance(element, ListConfig):
            for i, v in enumerate(element):
                mlflow.log_param(f'{param_name}.{i}', v)


def explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                explore_recursive(f'{parent_name}.{k}', v)
            else:
                mlflow.log_param(f'{parent_name}.{k}', v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f'{parent_name}.{i}', v)
