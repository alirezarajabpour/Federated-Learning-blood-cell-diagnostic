import torch
import mlflow
import numpy as np
import matplotlib.pyplot as plt

# Import your project's own modules
from dataset import load_data, prepare_client_datasets, get_client_dataloader
from model import get_model
from client import train, test  # We reuse the train and test functions

# --- Configuration ---
NUM_CLIENTS = 4  # Or 10, depending on your experiment
NUM_EPOCHS = 5   # Total training epochs for each isolated client
MLFLOW_TRACKING_URI = "http://mlflow:5000"


def run_siloed_training():
    """
    Trains and evaluates a separate model for each client and logs advanced metrics.
    """
    # Set a seed for reproducible model initialization
    torch.manual_seed(42)
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("MAP Federated Learning Showcase")

    # Load and partition the data once to ensure consistency
    train_dataset, _, num_classes = load_data()
    client_datasets, client_class_map = prepare_client_datasets(train_dataset, num_classes)

    print("--- Starting Siloed Training for All Clients ---")

    for client_id in range(NUM_CLIENTS):
        with mlflow.start_run(run_name=f"Siloed_Client_{client_id}"):
            print(f"\nTraining model for Client {client_id}...")

            mlflow.log_params({
                "strategy": "siloed", 
                "client_id": client_id,
                "epochs": NUM_EPOCHS
            })

            # Get this client's specific data
            trainloader, valloader = get_client_dataloader(client_id, client_datasets)
            if valloader is None:
                print(f"Client {client_id} has no data. Skipping.")
                continue

            available_classes = client_class_map[client_id]
            mlflow.log_param("available_classes", str(available_classes))

            # Each client gets a fresh, seeded model
            net = get_model(num_classes)
            hpm_model = get_model(num_classes)  # HPM is not used, but train function needs it
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Simple config for siloed training (only personalization phase, no HPM)
            config = {"epochs_agg": 0, "epochs_per": NUM_EPOCHS, "kd_lambda": 0.0}

            # Train the model ONLY on this client's data
            train(net, hpm_model, trainloader, available_classes, config, device)

            # --- MODIFIED: Evaluate the final model on the client's local validation set ---
            # The test function now returns a dictionary of metrics
            metrics = test(net, valloader, device, num_classes)
            
            accuracy = metrics["accuracy"]
            f1 = metrics["f1_score"]
            recall = metrics["recall"]
            
            print(f"  > Final Accuracy for Client {client_id}: {accuracy:.4f}")
            print(f"  > Final F1-Score for Client {client_id}: {f1:.4f}")

            # Log all the new metrics to MLflow
            mlflow.log_metric("final_local_accuracy", accuracy)
            mlflow.log_metric("final_local_f1_score", f1)
            mlflow.log_metric("final_local_recall", recall)

            # Log the confusion matrix as an artifact
            cm_fig = metrics["confusion_matrix_fig"]
            mlflow.log_figure(cm_fig, f"siloed_client_{client_id}_confusion_matrix.png")
            plt.close(cm_fig)


if __name__ == "__main__":
    run_siloed_training()