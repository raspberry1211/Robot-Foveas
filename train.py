import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from FovConvNeXt.models import make_model
import os

def main():
    print('Beginning training...')
    # Training parameters
    batch_size = 128
    num_epochs = 100
    learning_rate = 1e-3
    num_workers = 4
    n_fixations = 3  # Number of fixations for the active vision model
    
    # Model parameters
    radius = 0.5
    block_sigma = 0.1
    block_max_ord = 2
    patch_sigma = 0.1
    patch_max_ord = 2
    ds_sigma = 0.1
    ds_max_ord = 2
    
    # Data augmentation and normalization
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load ImageNet-100 dataset with different foveated versions
    train_datasets = []
    for version in ['X1', 'X2', 'X3', 'X4']:
        dataset = datasets.ImageFolder(
            root=f'data/imagenet-100/train.{version}',
            transform=train_transform
        )
        train_datasets.append(dataset)
    
    # Combine all training datasets
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    
    val_dataset = datasets.ImageFolder(
        root='data/imagenet-100/val.X',  # Using foveated validation set
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create model
    model = make_model(
        n_fixations=n_fixations,
        n_classes=100,  # ImageNet-100 has 100 classes
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
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
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
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # Print statistics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
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
                'val_loss': val_loss/len(val_loader),
                'train_acc': train_acc,
                'val_acc': val_acc
            }, f'checkpoint_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    main() 
