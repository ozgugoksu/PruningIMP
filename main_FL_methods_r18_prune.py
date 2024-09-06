#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: ozgu

IMP (iterative Magnitude Pruning) with Federated Learning on CIFAR10 dataset

"""

# Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import copy
import numpy as np
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
import random
from torchvision.models import resnet18

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define a simple CNN for CIFAR-10
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.model = resnet18(pretrained=False, num_classes=10)

    def forward(self, x):
        return self.model(x)

# Local training on client
def local_training(client_model, train_loader, epochs, lr):
    client_model.train()
    optimizer = optim.SGD(client_model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = client_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    return client_model

# Pruning function using Iterative Magnitude Pruning (IMP)
def prune_weights(model, prune_percentage):
    all_weights = torch.cat([torch.flatten(p.data) for p in model.parameters()])
    threshold = torch.quantile(torch.abs(all_weights), prune_percentage)

    for param in model.parameters():
        mask = torch.abs(param.data) > threshold
        param.data.mul_(mask)  # Zero out the pruned weights

    return model

# Federated averaging on the server
def federated_averaging(global_model, client_models):
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.mean(torch.stack([client.state_dict()[key] for client in client_models]), dim=0)

    global_model.load_state_dict(global_dict)
    return global_model

# Split dataset into non-IID for clients
def split_dataset(dataset, num_clients):
    total_size = len(dataset)
    sizes = [int(total_size / num_clients) for _ in range(num_clients)]
    client_datasets = random_split(dataset, sizes)
    return client_datasets

# Split dataset into IID for clients
def iid_partition(dataset, num_clients):
    data_per_client = len(dataset) // num_clients
    indices = np.random.permutation(len(dataset))
    return [Subset(dataset, indices[i * data_per_client:(i + 1) * data_per_client]) for i in range(num_clients)]

# Split dataset into non-IID for clients (version-2)
def non_iid_partition(dataset, num_clients, num_classes=10):
    class_indices = [[] for _ in range(num_classes)]
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    client_data = [[] for _ in range(num_clients)]
    
    for i in range(num_classes):
        random.shuffle(class_indices[i])
        split = np.array_split(class_indices[i], num_clients)
        for j in range(num_clients):
            client_data[j].extend(split[j])

    return [Subset(dataset, client_data[i]) for i in range(num_clients)]

# Evaluation of the global model
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)
    return accuracy, avg_loss

class FedAvg:
    def __init__(self, global_model, clients_data, device, prune_percentage=0.0):
        self.global_model = global_model
        self.clients_data = clients_data
        self.device = device
        self.prune_percentage = prune_percentage

    def local_training(self, client_model, train_loader, epochs, lr):
        client_model.train()
        client_model.to(self.device)
        optimizer = optim.SGD(client_model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = client_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        return client_model

    def aggregate(self, client_models):
        global_dict = self.global_model.state_dict()
        for key in global_dict.keys():
            # Average parameters
            global_dict[key] = torch.mean(torch.stack([client.state_dict()[key].float() for client in client_models]), dim=0)
        self.global_model.load_state_dict(global_dict)

    def train(self, num_rounds, local_epochs, lr, test_loader):
        round_accuracies = []
        round_losses = []

        for round in range(num_rounds):
            print(f"\nFedAvg Round {round+1}/{num_rounds}")
            client_models = []

            for i, client_data in enumerate(self.clients_data):
                print(f" Training on client {i+1}")
                client_model = copy.deepcopy(self.global_model)
                train_loader = DataLoader(client_data, batch_size=32, shuffle=True)
                client_model = self.local_training(client_model, train_loader, local_epochs, lr)
                
                if self.prune_percentage > 0:
                    client_model = prune_weights(client_model, self.prune_percentage)
                
                client_models.append(client_model)
            
            # Aggregate client models
            self.aggregate(client_models)
            
            # Evaluate global model
            accuracy, loss = evaluate_model(self.global_model, test_loader)
            round_accuracies.append(accuracy)
            round_losses.append(loss)
            print(f" Round {round+1} -- Test Accuracy: {accuracy:.2f}%, Test Loss: {loss:.4f}")
            
        # t-SNE Visualization
        visualize_tsne(self.global_model, test_loader)
        
        return round_accuracies, round_losses
    
    
# Plot function for accuracy and loss
def plot_performance(accuracies, losses, name):
    rounds = list(range(1, len(accuracies) + 1))
    
    plt.figure(figsize=(10, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(rounds, accuracies, marker='o', color='blue')
    plt.title('Federated Learning ' +  str(name) +': Test Accuracy per Round')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(rounds, losses, marker='o', color='red')
    plt.title('Federated Learning ' +  str(name) +': Test Loss per Round')
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


                


def visualize_tsne(model, data_loader):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            output = model(data)
            features.append(output.cpu().numpy())
            labels.append(target.cpu().numpy())

    features = np.concatenate(features)
    labels = np.concatenate(labels)

    tsne = TSNE(n_components=2, random_state=0)
    reduced_features = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title('t-SNE visualization of model features')
    plt.show()

               
# Example Usage
if __name__ == "__main__":
    # Transformations for the CIFAR-10 dataset
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    # Simulate federated data heterogeneity
    num_clients = 2
    # clients_data = split_dataset(train_dataset, num_clients)
    # clients_data = non_iid_partition(train_dataset, num_clients, num_classes=10)
    clients_data = iid_partition(train_dataset, num_clients)
    
    # Create test loader for evaluation
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize the global model (Simple CNN which is ResNet18)
    global_model = SimpleCNN()

    # Move model to GPU
    global_model = global_model.to(device) 

           
    # Define training parameters
    num_rounds = 4
    local_epochs = 5
    lr = 0.01
    prune_percentage = 0.2 # 20% pruning
    
    # Run federated learning with pruning and plot performance
    # Initialize FL algorithms

    # 1. FedAvg
    print("\n=== Starting FedAvg ===")

    fedavg = FedAvg(copy.deepcopy(global_model), clients_data, device, prune_percentage=prune_percentage)
    fedavg_acc, fedavg_loss = fedavg.train(num_rounds, local_epochs, lr, test_loader)
    plot_performance(fedavg_acc, fedavg_loss, "FedAvg")

    
    # # 2. FedProx
    # print("\n=== Starting FedProx ===")
    # fedprox = FedProx(copy.deepcopy(global_model), clients_data, device, mu=0.1, prune_percentage=prune_percentage)
    # fedprox_acc, fedprox_loss = fedprox.train(num_rounds, local_epochs, lr, test_loader)
    # plot_performance(fedprox_acc, fedprox_loss, "FedProx")
    
  