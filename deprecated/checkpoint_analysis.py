import torch
import os
import glob

def load_latest_checkpoint():
    # Find all checkpoint files
    checkpoint_files = glob.glob('train_2_checkpoint_epoch_*.pth')
    if not checkpoint_files:
        print("No checkpoint files found!")
        return None
    
    # Get the latest checkpoint based on epoch number
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(f"Loading checkpoint: {latest_checkpoint}")
    
    # Load the checkpoint
    checkpoint = torch.load(latest_checkpoint, map_location=torch.device('cpu'))
    
    # Print metrics
    print("\nModel Metrics:")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Train Loss: {checkpoint['train_loss']:.4f}")
    print(f"Validation Loss: {checkpoint['val_loss']:.4f}")
    print(f"Train Accuracy: {checkpoint['train_acc']:.2f}%")
    print(f"Validation Accuracy: {checkpoint['val_acc']:.2f}%")
    
    return checkpoint

if __name__ == "__main__":
    load_latest_checkpoint()
