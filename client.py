import threading
import time
import psutil
from prometheus_client import start_http_server, Gauge
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_MONITORING_ENABLED = True
except Exception as e:
    print(f"Warning: GPU monitoring is disabled. pynvml failed to initialize: {e}")
    GPU_MONITORING_ENABLED = False

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict
import argparse
import pickle
import numpy as np

from model import get_model
from dataset import load_data, prepare_client_datasets, get_client_dataloader, CLASS_NAMES

from sklearn.metrics import f1_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

CPU_GAUGE = Gauge('client_cpu_usage_percent', 'Current CPU usage of the client container.')
MEM_GAUGE = Gauge('client_memory_usage_percent', 'Current memory usage of the client container.')
if GPU_MONITORING_ENABLED:
    GPU_UTIL_GAUGE = Gauge('client_gpu_utilization_percent', 'Current GPU utilization.')
    GPU_MEM_GAUGE = Gauge('client_gpu_memory_percent', 'Current GPU memory usage.')


def monitor_resources(client_id):
    """A function that runs in a background thread to update metrics."""
    gpu_handle = None
    if GPU_MONITORING_ENABLED:
        try:

            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as e:
            print(f"[Client {client_id}] Could not get GPU handle: {e}")

    while True:
        CPU_GAUGE.set(psutil.cpu_percent(interval=1))
        MEM_GAUGE.set(psutil.virtual_memory().percent)
        if gpu_handle:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                GPU_UTIL_GAUGE.set(util.gpu)
                GPU_MEM_GAUGE.set(100 * mem.used / mem.total)
            except Exception as e:
                print(f"[Client {client_id}] Error getting GPU stats: {e}")
        time.sleep(5)  # Update every 5 seconds


def train(net, hpm_model, dataloader, available_classes, config, device):
    """Implements the two-stage training with epoch-level logging."""
    net.to(device)
    hpm_model.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    print("  > Starting Stage 1 (Aggregation)...")
    net.train()
    for epoch in range(config["epochs_agg"]):
        epoch_loss = []
        for images, labels in dataloader:
            images, labels = images.to(device), labels.squeeze().long().to(device)
            optimizer.zero_grad()
            logits = net(images)
            mask = torch.full_like(logits, -1e9)
            mask[:, available_classes] = 0
            masked_logits = logits + mask
            loss = criterion(masked_logits, labels)
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f"    - Aggregation Epoch {epoch + 1}/{config['epochs_agg']}, Avg. Loss: {np.mean(epoch_loss):.4f}")
    print("  > Starting Stage 2 (Personalization)...")
    kl_div_criterion = nn.KLDivLoss(reduction="batchmean")
    for epoch in range(config["epochs_per"]):
        epoch_loss = []
        for images, labels in dataloader:
            images, labels = images.to(device), labels.squeeze().long().to(device)
            optimizer.zero_grad()
            logits = net(images)
            with torch.no_grad():
                hpm_logits = hpm_model(images)
            loss_ce = criterion(logits, labels)
            loss_kd = kl_div_criterion(F.log_softmax(logits, dim=1), F.softmax(hpm_logits, dim=1))
            loss = loss_ce + config["kd_lambda"] * loss_kd
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f"    - Personalization Epoch {epoch + 1}/{config['epochs_per']}, Avg. Loss: {np.mean(epoch_loss):.4f}")

# scalable RS
# def train(net, hpm_model, dataloader, available_classes, config, device):
#     """Implements the two-stage training with the paper's alpha-scaling RS."""
#     net.to(device)
#     hpm_model.to(device)
#     optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
#     criterion = nn.CrossEntropyLoss()

#     # --- STAGE 1: AGGREGATION-FOCUSED TRAINING (with Alpha-Scaling RS) ---
#     print("  > Starting Stage 1 (Aggregation with Alpha-Scaling RS)...")
#     rs_alpha = config["rs_alpha"]
#     net.train()
#     for epoch in range(config["epochs_agg"]):
#         epoch_loss = []
#         for images, labels in dataloader:
#             images, labels = images.to(device), labels.squeeze().long().to(device)
#             optimizer.zero_grad()

#             logits = net(images)

#             # --- THIS IS THE CORRECTED "SOFT" RS IMPLEMENTATION ---
#             # Create a boolean mask of the classes this client has observed
#             observed_classes_mask = torch.zeros(logits.shape[1], dtype=torch.bool, device=device)
#             observed_classes_mask[available_classes] = True

#             # Apply the alpha scaling factor to the logits of unobserved classes
#             scaled_logits = logits.clone()  # Use clone to avoid modifying the original logits
#             scaled_logits[:, ~observed_classes_mask] *= rs_alpha

#             # Calculate loss on the scaled logits
#             loss = criterion(scaled_logits, labels)
#             # --- END OF CORRECTION ---

#             epoch_loss.append(loss.item())
#             loss.backward()
#             optimizer.step()
#         print(f"    - Aggregation Epoch {epoch + 1}/{config['epochs_agg']}, Avg. Loss: {np.mean(epoch_loss):.4f}")

#     # --- STAGE 2: PERSONALIZATION-FOCUSED TRAINING (This stage is unchanged) ---
#     print("  > Starting Stage 2 (Personalization)...")
#     kl_div_criterion = nn.KLDivLoss(reduction="batchmean")
#     for epoch in range(config["epochs_per"]):
#         epoch_loss = []
#         for images, labels in dataloader:
#             images, labels = images.to(device), labels.squeeze().long().to(device)
#             optimizer.zero_grad()
#             logits = net(images)
#             with torch.no_grad():
#                 hpm_logits = hpm_model(images)
#             loss_ce = criterion(logits, labels)
#             loss_kd = kl_div_criterion(F.log_softmax(logits, dim=1), F.softmax(hpm_logits, dim=1))
#             loss = loss_ce + config["kd_lambda"] * loss_kd
#             epoch_loss.append(loss.item())
#             loss.backward()
#             optimizer.step()
#         print(f"    - Personalization Epoch {epoch + 1}/{config['epochs_per']}, Avg. Loss: {np.mean(epoch_loss):.4f}")


# def test(model, dataloader, device):
#     model.to(device)
#     model.eval()
#     criterion = nn.CrossEntropyLoss()
#     correct, total, loss = 0, 0, 0.0
#     with torch.no_grad():
#         for images, labels in dataloader:
#             images, labels = images.to(device), labels.squeeze().long().to(device)
#             outputs = model(images)
#             loss += criterion(outputs, labels).item()
#             total += labels.size(0)
#             correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
#     return loss / len(dataloader), correct / total

def test(model, dataloader, device, num_classes):
    """Evaluates the model and returns a dictionary of advanced metrics."""
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_labels = []
    all_preds = []
    loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.squeeze().long().to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_loss = loss / len(dataloader)
    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=CLASS_NAMES.values(), yticklabels=CLASS_NAMES.values())
    ax.set(xlabel='Predicted Label', ylabel='True Label', title='Confusion Matrix')

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1_score": f1,
        "recall": recall,
        "confusion_matrix_fig": fig
    }
    return metrics


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, hpm_model, trainloader, valloader, available_classes, num_classes):
        self.cid = cid
        self.net = net
        self.hpm_model = hpm_model
        self.trainloader = trainloader
        self.valloader = valloader
        self.available_classes = available_classes
        self.num_classes = num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[Client {self.cid}] Initialized with {len(self.trainloader.dataset)} training samples.")

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """This now only sets the main network's parameters."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    # def fit(self, parameters, config):
    #     self.set_parameters(parameters)
    #     hpm_state_numpy = config.get("hpm_state")
    #     if hpm_state_numpy is not None:
    #         hpm_state = OrderedDict({k: torch.tensor(v) for k, v in zip(self.hpm_model.state_dict().keys(), hpm_state_numpy)})
    #         self.hpm_model.load_state_dict(hpm_state, strict=True)
    #     else:
    #         self.hpm_model.load_state_dict(self.net.state_dict(), strict=True)

    #     print(f"Client {self.cid} training on {len(self.available_classes)} classes: {self.available_classes}")
    #     train(self.net, self.hpm_model, self.trainloader, self.available_classes, config, self.device)
    #     print(f"  > Client {self.cid} training complete.")

    #     p_k_t_state_dict = [val.cpu().numpy() for _, val in self.net.state_dict().items()]
    #     p_k_t_state_dict_bytes = pickle.dumps(p_k_t_state_dict)
    #     # The client TELLS the server its ID in the metrics
    #     metrics = {"p_k_t_state_dict": p_k_t_state_dict_bytes, "logical_id": self.cid}
    #     return self.get_parameters({}), len(self.trainloader.dataset), metrics

    # def evaluate(self, parameters, config):
    #     self.set_parameters(parameters)
    #     hpm_state_numpy = config.get("hpm_state")
    #     if hpm_state_numpy is not None:
    #         hpm_state = OrderedDict({k: torch.tensor(v) for k, v in zip(self.hpm_model.state_dict().keys(), hpm_state_numpy)})
    #         self.hpm_model.load_state_dict(hpm_state, strict=True)

    #     loss_personalized, acc_personalized = test(self.net, self.valloader, self.device)
    #     loss_hpm, acc_hpm = test(self.hpm_model, self.valloader, self.device)

    #     return float(loss_personalized), len(self.valloader.dataset), {
    #         "accuracy_personalized": float(acc_personalized),
    #         "accuracy_hpm": float(acc_hpm),
    #         "logical_id": self.cid,  # Also report ID during evaluation
    #     }

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        hpm_state_numpy = config.get("hpm_state")
        if hpm_state_numpy is not None:
            hpm_state = OrderedDict({k: torch.tensor(v) for k, v in zip(self.hpm_model.state_dict().keys(), hpm_state_numpy)})
            self.hpm_model.load_state_dict(hpm_state, strict=True)
        else:
            self.hpm_model.load_state_dict(self.net.state_dict(), strict=True)

        print(f"Client {self.cid} training on {len(self.available_classes)} classes: {self.available_classes}")
        train(self.net, self.hpm_model, self.trainloader, self.available_classes, config, self.device)
        print(f"  > Client {self.cid} training complete.")

        p_k_t_state_dict = [val.cpu().numpy() for _, val in self.net.state_dict().items()]
        p_k_t_state_dict_bytes = pickle.dumps(p_k_t_state_dict)

        metrics = {
            "p_k_t_state_dict": p_k_t_state_dict_bytes, 
            "logical_id": self.cid,
            "train_samples": len(self.trainloader.dataset)
        }
        return self.get_parameters({}), len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        hpm_state_numpy = config.get("hpm_state")
        if hpm_state_numpy is not None:
            hpm_state = OrderedDict({k: torch.tensor(v) for k, v in zip(self.hpm_model.state_dict().keys(), hpm_state_numpy)})
            self.hpm_model.load_state_dict(hpm_state, strict=True)

        metrics_personalized = test(self.net, self.valloader, self.device, self.num_classes)
        metrics_hpm = test(self.hpm_model, self.valloader, self.device, self.num_classes)

        plt.close(metrics_personalized["confusion_matrix_fig"])
        plt.close(metrics_hpm["confusion_matrix_fig"])

        final_metrics = {
            "accuracy_personalized": metrics_personalized["accuracy"],
            "f1_personalized": metrics_personalized["f1_score"],
            "recall_personalized": metrics_personalized["recall"],
            "accuracy_hpm": metrics_hpm["accuracy"],
            "logical_id": self.cid,
        }

        return float(metrics_personalized["loss"]), len(self.valloader.dataset), final_metrics


def main():
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(description="Flower Client for Advanced MAP")
    parser.add_argument("--client-id", type=int, required=True, help="Client ID (from 0 to 9)")
    parser.add_argument("--server-address", type=str, default="server:8080", help="Address of the Flower server")
    args = parser.parse_args()

    metrics_port = 8001 + args.client_id
    start_http_server(metrics_port)

    monitor_thread = threading.Thread(target=monitor_resources, args=(args.client_id,), daemon=True)
    monitor_thread.start()
    print(f"[Client {args.client_id}] Prometheus metrics server started on port {metrics_port}.")
    # ----------------------------------------

    train_dataset, _, num_classes = load_data()
    client_datasets, client_class_map = prepare_client_datasets(train_dataset, num_classes)

    client_id = args.client_id

    trainloader, valloader = get_client_dataloader(client_id, client_datasets)

    if trainloader is None or valloader is None:
        print(f"Client {client_id} has no data. Exiting.")
        return

    available_classes = client_class_map[client_id]
    net = get_model(num_classes)
    hpm_model = get_model(num_classes)

    client = FlowerClient(str(client_id), net, hpm_model, trainloader, valloader, available_classes, num_classes)
    fl.client.start_client(server_address=args.server_address, client=client.to_client())


if __name__ == "__main__":
    main()
