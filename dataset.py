import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from medmnist import BloodMNIST
import numpy as np

NUM_CLIENTS = 4
DATA_ROOT = "./data"
DATA_SEED = 43

CLASS_NAMES = {
    0: 'basophil', 1: 'eosinophil', 2: 'erythroblast',
    3: 'ig', 4: 'lymphocyte', 5: 'monocyte',
    6: 'neutrophil', 7: 'platelet'
}


def load_data():
    """Loads the BloodMNIST dataset and returns train, test sets and number of classes."""

    data_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    train_dataset = BloodMNIST(split="train", download=True, transform=data_transforms, root=DATA_ROOT)
    test_dataset = BloodMNIST(split="test", download=True, transform=data_transforms, root=DATA_ROOT)

    num_classes = len(train_dataset.info['label'])
    return train_dataset, test_dataset, num_classes


def prepare_client_datasets(train_dataset, num_classes):
    """
    Partitions the dataset to simulate a federated environment where each
    client is missing some classes.
    """
    print("Partitioning data to create the 'missing classes' scenario...")
    all_labels = np.array([label[0] for label in train_dataset.labels])
    client_data_indices = [[] for _ in range(NUM_CLIENTS)]
    client_available_classes = [set() for _ in range(NUM_CLIENTS)]

    rng = np.random.default_rng(DATA_SEED)

    for k in range(num_classes):
        idx_k = np.where(all_labels == k)[0]
        rng.shuffle(idx_k)

        num_clients_for_class = int(0.7 * NUM_CLIENTS)
        clients_for_class = rng.choice(range(NUM_CLIENTS), num_clients_for_class, replace=False)

        split_size = len(idx_k) // num_clients_for_class
        for i, client_id in enumerate(clients_for_class):
            start = i * split_size
            end = (i + 1) * split_size if i < num_clients_for_class - 1 else len(idx_k)
            client_data_indices[client_id].extend(idx_k[start:end])
            client_available_classes[client_id].add(k)

    client_available_classes = {i: sorted(list(s)) for i, s in enumerate(client_available_classes)}
    client_datasets = [Subset(train_dataset, indices) for indices in client_data_indices]

    print("Data partitioning complete.")
    return client_datasets, client_available_classes


def get_client_dataloader(client_id, client_datasets, batch_size=32):
    """
    Returns a training dataloader and a validation dataloader for a specific client.
    The client's data is split into 80% for training and 20% for validation.
    """
    client_dataset = client_datasets[client_id]
    if not client_dataset:
        return None, None

    # len_val = len(client_dataset) // 5
    # len_train = len(client_dataset) - len_val

    len_val = max(1, len(client_dataset) // 5)  # At least 1, or 20%
    len_train = len(client_dataset) - len_val
    # -----------------------------

    ds_train, ds_val = random_split(client_dataset, [len_train, len_val], generator=torch.Generator().manual_seed(42))

    trainloader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    valloader = DataLoader(ds_val, batch_size=batch_size)

    return trainloader, valloader
