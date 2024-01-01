import mlflow
import os
import hydra
from omegaconf import DictConfig, OmegaConf


# This automatically reads in the configuration
@hydra.main(config_path='.', config_name='config', version_base=None)
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    # Check which steps we need to execute
    if isinstance(config["main"]["execute_steps"], str):
        # This was passed on the command line as a comma-separated list of steps
        steps_to_execute = config["main"]["execute_steps"].split(",")
    else:
        print(config["main"]["execute_steps"])
        print(type(config["main"]["execute_steps"]))
        # assert isinstance(config["main"]["execute_steps"], list)
        steps_to_execute = config["main"]["execute_steps"]

    # Download step
    if "download" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "download"),
            "main",
            parameters={
                "file_url": config["data"]["file_url"],
                "artifact_name": "raw_data",
                "artifact_type": "raw_data",
                "artifact_description": "Data as downloaded"
            },
        )

    if "preprocess" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "preprocess"),
            "main",
            parameters={
                "input_artifact": "raw_data:latest",
                "artifact_name": "preprocessed_data",
                "artifact_type": "preprocessed_data",
                "artifact_description": "Preprocessed data"
            },
        )

    # if "check_data" in steps_to_execute:
    #     _ = mlflow.run(
    #         os.path.join(root_path, "check_data"),
    #         "main",
    #         parameters={
    #             "ks_alpha": "0.05",
    #             "sample_artifact": "preprocessed_data:latest",
    #             "reference_artifact": config["data"]["reference_dataset"]
    #         },
    #     )

    if "segregate" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "segregate"),
            "main",
            parameters={
                "input_artifact": "preprocessed_data:latest",
                "artifact_root": "segregated_data",
                "artifact_type": "segregated_data",
                "test_size": config["data"]["test_size"]
            },
        )

    if "random_forest" in steps_to_execute:

        # Serialize decision tree configuration
        model_config = os.path.abspath("random_forest_config.yml")

        with open(model_config, "w+") as fp:
            fp.write(OmegaConf.to_yaml(config["random_forest_pipeline"]))

        _ = mlflow.run(
            os.path.join(root_path, "random_forest"),
            "main",
            parameters={
                "train_data": "segregated_data_train.csv:latest",
                "model_config": model_config,
                "export_artifact": config["random_forest_pipeline"]["export_artifact"],
                "stratify": config["data"]["stratify"]
            },
        )

    if "evaluate" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "evaluate"),
            "main",
            parameters={
                "test_data": "segregated_data_test.csv:latest",
                "model_export": f"{config['random_forest_pipeline']['export_artifact']}:latest",
            },
        )


if __name__ == "__main__":
    go()
