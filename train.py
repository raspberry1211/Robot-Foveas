import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, Subset, DataLoader
from torchvision import datasets, transforms
from torchvision import models
from FovConvNeXt.models import make_model
import numpy as np
import os

def main():

    # Print messages to let the user know how it's progressing
    print('Beginning training...')
    
    # Training parameters
    n_classes = 100 # Imagenet-100 has 100 classes
    batch_size = 64
    num_epochs = 200
    learning_rate = 0.004
    num_workers = 4
    n_fixations = 1  # Number of fixations for the active vision model
    max_grad_norm = 1.0  # Gradient clipping threshold
    weight_decay = 0.005
    
    # Model parameters
    radius = 0.4
    block_sigma = 0.8
    block_max_ord = 4
    patch_sigma = 1.0
    patch_max_ord = 4
    ds_sigma = 0.6
    ds_max_ord = 0

    print('Loading data')
    
    # Dataset paths
    DATA_PATH = "/rhome/drfj2024/Robot-Foveas/data/imagenet-100/train.X" # Path to all the training images. This later gets split into train & test sets
    VAL_PATH = "/rhome/drfj2024/Robot-Foveas/data/imagenet-100/val.X"

    # Training transforms
    train_transform = transforms.Compose([
        transforms.TrivialAugmentWide(),
        transforms.Resize(256),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation & Test transforms
    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the full dataset
    full_dataset = datasets.ImageFolder(DATA_PATH)

    def split_dataset(dataset, train_size):
        """Split the dataset into training and test"""
        total_size = len(dataset)
        indices = np.arange(total_size)
        np.random.shuffle(indices)
        train_idx = int(total_size * train_size)
        
        train_indices = indices[:train_idx]
        test_indices = indices[train_idx:]
        
        return train_indices, test_indices

    # Custom dataset class to apply different transforms
    class TransformDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform
        
        def __getitem__(self, index):
            x, y = self.dataset[index]
            return self.transform(x), y
        
        def __len__(self):
            return len(self.dataset)

    # Split the training dataset
    train_indices, test_indices = split_dataset(
        combined_train_dataset,
        train_size=0.8, # 80% of dataset is used for training, 20% for testing
    )

    # Create subsets with appropriate transforms
    train_dataset = TransformDataset(
        Subset(combined_train_dataset, train_indices),
        train_transform
    )

    test_dataset = TransformDataset(
        Subset(combined_train_dataset, test_indices),
        val_test_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    print('Data loaded')

    
    # Create model with correct number of classes
    model = make_model(
        n_fixations=n_fixations,
        n_classes=n_classes, 
        radius=radius,
        block_sigma=block_sigma,
        block_max_ord=block_max_ord,
        patch_sigma=patch_sigma,
        patch_max_ord=patch_max_ord,
        ds_sigma=ds_sigma,
        ds_max_ord=ds_max_ord
    )
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss function and optimizer with increased weight decay
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Early stopping parameters
    patience = 20
    best_test_acc = 0
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            if torch.isnan(loss):
                print("Warning: NaN loss detected, skipping batch")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        # Testing
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            # for inputs, targets in test_loader:
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        # Print statistics
        train_acc = 100. * train_correct / train_total
        test_acc = 100. * test_correct / test_total
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Acc: {test_acc:.2f}%')
        
        # Early stopping
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss/len(train_loader),
                'test_loss': test_loss/len(test_loader),
                'train_acc': train_acc,
                'test_acc': test_acc
            }, 'fixed_settings_best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss/len(train_loader),
                'test_loss': test_loss/len(test_loader),
                'train_acc': train_acc,
                'test_acc': test_acc
            }, f'fixed_settings_checkpoint_epoch_{epoch+1}.pth')

    # Once training is complete, validate
    # Load validation dataset
    val_dataset = datasets.ImageFolder(VAL_PATH)

    # Apply transform to validation dataset
    val_dataset = TransformDataset(val_dataset, val_test_transform)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
        
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
                
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

    val_acc = 100. * val_correct / val_total
    print(f'Final Validation Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')

    
# Run training function
if __name__ == '__main__':
    main() 
