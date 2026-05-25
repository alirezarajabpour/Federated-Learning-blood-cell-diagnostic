import flwr as fl
from flwr.common import ndarrays_to_parameters
from typing import Dict, Optional, Tuple
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
import mlflow
import argparse
from flwr.server.strategy import FedAvg
import json
import matplotlib.pyplot as plt

from model import get_model
from dataset import load_data, prepare_client_datasets, CLASS_NAMES
from strategy import MapStrategy
from client import test

MODEL_PATH = "final_global_model.pth"

HPM_BETA = 0.3
KD_LAMBDA = 0.1
RS_ALPHA = 0.9
EPOCHS_AGG = 4
EPOCHS_PER = 4
NUM_ROUNDS = 10
NUM_CLIENTS = 4


def fit_config(server_round: int):
    return {
        "server_round": server_round,
        "kd_lambda": KD_LAMBDA,
        "epochs_agg": EPOCHS_AGG,
        "epochs_per": EPOCHS_PER,
        "rs_alpha": RS_ALPHA,
    }


# def get_evaluate_fn(model, test_dataset):
#     def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
#         params_dict = zip(model.state_dict().keys(), parameters)
#         state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#         model.load_state_dict(state_dict, strict=True)
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         testloader = DataLoader(test_dataset, batch_size=64)
#         loss, accuracy = test(model, testloader, device)
#         print(f"Server-side evaluation round {server_round} - Global Model Accuracy: {accuracy:.4f}")
#         return loss, {"global_accuracy": accuracy}
#     return evaluate


def get_evaluate_fn(model, test_dataset):
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        num_classes = len(CLASS_NAMES)  # Get num_classes from dataset
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        testloader = DataLoader(test_dataset, batch_size=64)

        metrics = test(model, testloader, device, num_classes)

        loss = metrics["loss"]
        accuracy = metrics["accuracy"]
        f1 = metrics["f1_score"]
        recall = metrics["recall"]
        cm_fig = metrics["confusion_matrix_fig"]

        print(f"Server-side evaluation round {server_round} - Global Model Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

        # mlflow.log_metric("centralized_global_accuracy_server", accuracy, step=server_round)
        # mlflow.log_metric("centralized_global_f1_score_server", f1, step=server_round)
        # mlflow.log_metric("centralized_global_recall_server", recall, step=server_round)

        mlflow.log_figure(cm_fig, f"confusion_matrices/cm_round_{server_round}.png")
        plt.close(cm_fig)

        return loss, {"global_accuracy": accuracy, "global_f1_score": f1, "global_recall": recall}
    return evaluate


def main():
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(description="Flower Server")
    parser.add_argument(
        "--strategy", type=str, choices=["fedavg", "map"], default="map",
        help="Which federated strategy to use"
    )
    args = parser.parse_args()
    print(f"--- Starting server with '{args.strategy}' strategy ---")

    train_dataset, test_dataset, num_classes = load_data()
    _, client_class_map = prepare_client_datasets(train_dataset, num_classes)

    model = get_model(num_classes)
    initial_parameters = ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in model.state_dict().items()]
    )

    # strategy = MapStrategy(
    #     fraction_fit=1.0,
    #     fraction_evaluate=1.0,
    #     min_fit_clients=NUM_CLIENTS,
    #     min_evaluate_clients=NUM_CLIENTS,
    #     min_available_clients=NUM_CLIENTS,
    #     on_fit_config_fn=fit_config,
    #     evaluate_fn=get_evaluate_fn(model, test_dataset),
    #     hpm_beta=HPM_BETA,
    #     initial_parameters=initial_parameters,
    # )

    if args.strategy == "map":
        strategy = MapStrategy(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=NUM_CLIENTS,
            min_evaluate_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
            on_fit_config_fn=fit_config,
            evaluate_fn=get_evaluate_fn(model, test_dataset),
            hpm_beta=HPM_BETA,
            initial_parameters=initial_parameters,
        )
    else:  # fedavg
        strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=NUM_CLIENTS,
            min_evaluate_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
            on_fit_config_fn=fit_config,
            evaluate_fn=get_evaluate_fn(model, test_dataset),
            initial_parameters=initial_parameters,
        )
    # -----------------------------------------------

    # --- MLflow Integration ---
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("MAP Federated Learning")

    with mlflow.start_run(run_name=f"Run_{args.strategy.upper()}"):
        print("Starting MLflow Run...")

        mlflow.log_params({
            "strategy": args.strategy,
            "num_clients": NUM_CLIENTS,
            "num_rounds": NUM_ROUNDS,
            "hpm_beta": HPM_BETA,
            "kd_lambda": KD_LAMBDA,
            "epochs_agg": EPOCHS_AGG,
            "epochs_per": EPOCHS_PER,
            "rs_alpha": RS_ALPHA,
        })

        client_class_map_str = json.dumps(client_class_map, indent=2)
        mlflow.log_param("client_class_distribution", client_class_map_str)
        # -------------------------------------------

        history = fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=strategy,
        )

        print("Logging final metrics to MLflow...")
        for metric_name, values in history.metrics_centralized.items():
            for server_round, value in values:
                mlflow.log_metric(f"centralized_{metric_name}", value, step=server_round)

        for metric_name, values in history.metrics_distributed.items():
            for server_round, value in values:
                mlflow.log_metric(f"distributed_{metric_name}", value, step=server_round)

        # --- Save the final global model ---
        if args.strategy == "map" and history.losses_centralized:
            print("Saving final global model...")

            final_parameters_ndarrays = fl.common.parameters_to_ndarrays(strategy.final_parameters)  # Create a state dictionary

            final_state_dict = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), final_parameters_ndarrays)}

            torch.save(final_state_dict, MODEL_PATH)
            print(f"Final global model saved to {MODEL_PATH}")
            mlflow.log_artifact(MODEL_PATH)
        # ------------------------------------

        print("MLflow Run Finished.")


if __name__ == "__main__":
    main()
