import mlflow
from .utils import ConfigLoader

class MLFlowManager:
    """
    A class to manage MLflow tracking and experiment setup.
    """

    def __init__(self):
        """
        Initializes the MLFlowManager.
        """
        self.configs = ConfigLoader()


    def setup_mlflow(self):
        """
        Configures MLflow tracking URI and sets the experiment.
        """
        try:
            tracking_uri = self.configs.get("MLFLOW_SERVER_URL", None)
            experiment_name = self.configs.get("MLFLOW_EXPERIMENT_NAME", "DefaultExperiment")

            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)

            mlflow.set_experiment(experiment_name)
            mlflow.dspy.autolog()

            print(f"MLflow initialized with experiment: {experiment_name}")

        except Exception as e:
            print(f"Error initializing MLflow: {e}")

    def set_experiment(self, experiment_name: str):
        """
        Updates the MLflow experiment name.

        :param experiment_name: The name of the new MLflow experiment.
        """
        try:
            mlflow.set_experiment(experiment_name)
            print(f"MLflow experiment set to: {experiment_name}")
        except Exception as e:
            print(f"Error setting MLflow experiment: {e}")

    def get_experiment(self):
        """
        Retrieves the active MLflow experiment details.
        """
        try:
            experiment = mlflow.get_experiment_by_name(mlflow.get_experiment().name)
            return experiment
        except Exception as e:
            print(f"Error retrieving MLflow experiment: {e}")
            return None
