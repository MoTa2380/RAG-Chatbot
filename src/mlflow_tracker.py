import mlflow

def start_experiment(experiment_name: str):
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run()
    return run

def log_metric(name: str, value: float):
    mlflow.log_metric(name, value)

def log_param(name: str, value):
    mlflow.log_param(name, value)

def end_experiment():
    mlflow.end_run()
