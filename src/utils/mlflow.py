import mlflow
from omegaconf import DictConfig, ListConfig


def log_params_to_mlflow(params):
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


def search_run(run_name):
    runs = mlflow.search_runs()
    run = runs[runs['tags.mlflow.runName'] == run_name].reset_index(drop=True)

    if len(run) == 0:
        return None
    elif len(run) == 1:
        return run.loc[0, 'run_id']
    else:
        raise ValueError('Duplicate runs exist. ' + str(run))
