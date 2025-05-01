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
    learning_rate = 1e-4  # Reduced from 5e-4
    num_workers = 4
    n_fixations = 1  # Number of fixations for the active vision model
    max_grad_norm = 1.0  # Gradient clipping threshold
    weight_decay = 0.05  # Increased from 0.01
    
    # Model parameters
    radius = 0.6
    block_sigma = 0.05
    block_max_ord = 2
    patch_sigma = 0.05
    patch_max_ord = 2
    ds_sigma = 0.05
    ds_max_ord = 2
    
    # Enhanced data augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
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
        print(f'Loaded training dataset {version} with {len(dataset)} samples')
    
    # Combine all training datasets
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    print(f'Total training samples: {len(train_dataset)}')
    
    val_dataset = datasets.ImageFolder(
        root='data/imagenet-100/val.X',  # Using foveated validation set
        transform=val_transform
    )
    print(f'Validation samples: {len(val_dataset)}')
    
    # Verify data loading
    print('\nVerifying data loading...')
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
    train_sample, train_label = next(iter(train_loader))
    val_sample, val_label = next(iter(val_loader))
    print(f'Training batch shape: {train_sample.shape}')
    print(f'Validation batch shape: {val_sample.shape}')
    print(f'Training label range: {train_label.min()} to {train_label.max()}')
    print(f'Validation label range: {val_label.min()} to {val_label.max()}')
    
    # Check for duplicate samples
    train_paths = set()
    for dataset in train_datasets:
        for path, _ in dataset.samples:
            train_paths.add(path)
    val_paths = set(path for path, _ in val_dataset.samples)
    duplicates = train_paths.intersection(val_paths)
    if duplicates:
        print(f'Warning: Found {len(duplicates)} duplicate samples between train and val sets')
    
    # Create model with reduced complexity
    model = make_model(
        n_fixations=n_fixations,
        n_classes=100,
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
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)  # Increased from 0.1
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Early stopping parameters
    patience = 5
    best_val_acc = 0
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
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
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
                'val_loss': val_loss/len(val_loader),
                'train_acc': train_acc,
                'val_acc': val_acc
            }, f'train_2_checkpoint_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    main() 
