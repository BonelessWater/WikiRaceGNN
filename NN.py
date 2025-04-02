import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from torch.cuda.amp import GradScaler, autocast

# Define an Input Transformation Module
# It outputs a small "delta" which is added to the original input.
class InputTransformer(nn.Module):
    def __init__(self, input_shape=(1, 28, 28)):
        super(InputTransformer, self).__init__()
        # A small network that outputs a correction for each pixel
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1)
        )
        # Optionally, you could add a scaling factor to limit the magnitude of changes
        self.scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        delta = self.conv(x)
        # The transformation is a small adjustment added to the input
        return x + self.scale * delta

# Define the Classifier Module that predicts a binary result.
class Classifier(nn.Module):
    def __init__(self, input_size=784, hidden_size=256):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)  # Single output for binary classification
        self.hardsigmoid = nn.Hardsigmoid()  # Fast alternative to sigmoid
     
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.hardsigmoid(x)    # Ensure output is in [0,1]
        return x

# Combine the two modules into one network.
class FullModel(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), hidden_size=256):
        super(FullModel, self).__init__()
        self.transformer = InputTransformer(input_shape=input_shape)
        self.classifier = Classifier(input_size=input_shape[0]*input_shape[1]*input_shape[2], 
                                     hidden_size=hidden_size)
    
    def forward(self, x):
        # First modify the input
        x_transformed = self.transformer(x)
        # Then classify the modified input
        out = self.classifier(x_transformed)
        return out, x_transformed  # Optionally, return the modified input for analysis

def main_worker(gpu, ngpus):
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")

    if ngpus > 1:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=ngpus, rank=gpu)

    # Create a synthetic binary classification dataset
    num_samples = 60000
    input_shape = (1, 28, 28)
    # Synthetic inputs (e.g., images) and binary labels (0 or 1)
    X = torch.randn(num_samples, *input_shape)
    y = torch.randint(0, 2, (num_samples, 1)).float()  # float labels for BCELoss
    dataset = TensorDataset(X, y)
    
    sampler = DistributedSampler(dataset, num_replicas=ngpus, rank=gpu, shuffle=True) if ngpus > 1 else None
    data_loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=(sampler is None),
        num_workers=4,
        pin_memory=True,
        sampler=sampler
    )

    # Initialize the full model and move it to the device
    model = FullModel(input_shape=input_shape, hidden_size=256).to(device)
    if ngpus > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    
    # Loss function and optimizer
    # We add an extra penalty term to discourage large modifications to the input.
    # This is optional but can help keep the changes subtle.
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()

    num_epochs = 5
    model.train()
    start_time = time.time()
    for epoch in range(num_epochs):
        if sampler:
            sampler.set_epoch(epoch)
        epoch_loss = 0.0
        for inputs, labels in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with autocast():
                outputs, transformed = model(inputs)
                # Primary classification loss
                loss = criterion(outputs, labels)
                # Add a penalty for large modifications to the input
                modification_loss = torch.mean((transformed - inputs)**2)
                total_loss = loss + 0.01 * modification_loss
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += total_loss.item() * inputs.size(0)
        
        if gpu == 0:
            avg_loss = epoch_loss / num_samples
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    if gpu == 0:
        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds.")

    if ngpus > 1:
        dist.destroy_process_group()

def main():
    ngpus = torch.cuda.device_count()
    if ngpus > 1:
        print(f"Using DistributedDataParallel with {ngpus} GPUs for faster training.")
        mp.spawn(main_worker, nprocs=ngpus, args=(ngpus,))
    else:
        print("Using single GPU (or CPU) for training.")
        main_worker(0, ngpus)

if __name__ == '__main__':
    main()
