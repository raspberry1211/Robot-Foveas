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
    class_mapping = {}  # Maps (version, class) to global class index
    current_global_class = 0
    
    for version_idx, version in enumerate(['X1', 'X2', 'X3', 'X4']):
        dataset = datasets.ImageFolder(
            root=f'data/imagenet-100/train.{version}',
            transform=train_transform
        )
        
        # Create mapping for this version's classes
        version_classes = set()
        for _, label in dataset:
            version_classes.add(label)
        
        # Map local classes to global classes
        for local_class in sorted(version_classes):
            class_mapping[(version, local_class)] = current_global_class
            current_global_class += 1
        
        print(f'Loaded training dataset {version} with {len(dataset)} samples')
        train_datasets.append(dataset)
    
    # Create a wrapper dataset that applies the class mapping
    class MappedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, version, class_mapping):
            self.dataset = dataset
            self.version = version
            self.class_mapping = class_mapping
            
        def __getitem__(self, idx):
            img, label = self.dataset[idx]
            if self.version == 'X':  # Validation set - use direct mapping
                global_label = label
            else:  # Training set - use version-specific mapping
                global_label = self.class_mapping[(self.version, label)]
            return img, global_label
            
        def __len__(self):
            return len(self.dataset)
    
    # Apply class mapping to each dataset
    mapped_datasets = []
    for version, dataset in zip(['X1', 'X2', 'X3', 'X4'], train_datasets):
        mapped_datasets.append(MappedDataset(dataset, version, class_mapping))
    
    # Combine all training datasets
    train_dataset = torch.utils.data.ConcatDataset(mapped_datasets)
    print(f'Total training samples: {len(train_dataset)}')
    
    # Load validation set
    val_dataset = datasets.ImageFolder(
        root='data/imagenet-100/val.X',
        transform=val_transform
    )
    
    # Create mapped validation dataset
    val_dataset = MappedDataset(val_dataset, 'X', class_mapping)
    print(f'Validation samples: {len(val_dataset)}')
    
    # # Detailed dataset verification
    # print('\nDetailed Dataset Verification:')
    
    # # Check class distribution in training set
    # train_class_counts = {}
    # for dataset in mapped_datasets:
    #     for _, label in dataset:
    #         train_class_counts[label] = train_class_counts.get(label, 0) + 1
    # print('\nTraining set class distribution:')
    # for label, count in sorted(train_class_counts.items()):
    #     print(f'Class {label}: {count} samples')
    
    # # Check class distribution in validation set
    # val_class_counts = {}
    # for _, label in val_dataset:
    #     val_class_counts[label] = val_class_counts.get(label, 0) + 1
    # print('\nValidation set class distribution:')
    # for label, count in sorted(val_class_counts.items()):
    #     print(f'Class {label}: {count} samples')
    
    # # Check for missing classes
    # train_classes = set(train_class_counts.keys())
    # val_classes = set(val_class_counts.keys())
    # missing_in_val = train_classes - val_classes
    # missing_in_train = val_classes - train_classes
    
    # if missing_in_val:
    #     print(f'\nWarning: {len(missing_in_val)} classes present in training but missing in validation:')
    #     print(sorted(missing_in_val))
    # if missing_in_train:
    #     print(f'\nWarning: {len(missing_in_train)} classes present in validation but missing in training:')
    #     print(sorted(missing_in_train))
    
    # # Check data paths
    # print('\nChecking data paths:')
    # print(f'Training path: data/imagenet-100/train.X1')
    # print(f'Validation path: data/imagenet-100/val.X')
    
    # Create data loaders
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
    
    # Verify data loading
    print('\nVerifying data loading...')
    train_sample, train_label = next(iter(train_loader))
    val_sample, val_label = next(iter(val_loader))
    print(f'Training batch shape: {train_sample.shape}')
    print(f'Validation batch shape: {val_sample.shape}')
    print(f'Training label range: {train_label.min()} to {train_label.max()}')
    print(f'Validation label range: {val_label.min()} to {val_label.max()}')
    
    # Get actual number of classes
    n_classes = 100  # We know it's ImageNet-100
    print(f'\nUsing {n_classes} classes for model')
    
    # # Check for duplicate samples
    # train_paths = set()
    # for dataset in mapped_datasets:
    #     for path, _ in dataset.dataset.samples:
    #         train_paths.add(path)
    # val_paths = set(path for path, _ in val_dataset.dataset.samples)
    # duplicates = train_paths.intersection(val_paths)
    # if duplicates:
    #     print(f'Warning: Found {len(duplicates)} duplicate samples between train and val sets')
    
    # Create model with correct number of classes
    model = make_model(
        n_fixations=n_fixations,
        n_classes=n_classes,  # Use full 100 classes
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
