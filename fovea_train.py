import os
import time
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from FovConvNeXt.models import make_model

# ==== Hyperparameters ====
class Params:
    def __init__(self):
        self.batch_size = 64
        self.epochs = 90
        self.lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.lr_step_size = 30
        self.lr_gamma = 0.1
        self.workers = 4
        self.num_classes = 100
        self.name = "fovea_imagenet100"
        self.resume_checkpoint = "checkpoints/fovea_imagenet100/checkpoint_epoch57.pth"

params = Params()

# ==== Device ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# ==== Transforms ====
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

# ==== Load Training Data from 4 Folders ====
train_dirs = ["train.X1", "train.X2", "train.X3", "train.X4"]

# Get consistent class-to-index mapping from val.X
global_class_names = sorted(os.listdir("val.X"))
global_class_to_idx = {cls_name: idx for idx, cls_name in enumerate(global_class_names)}

from glob import glob

train_datasets = []
for train_dir in train_dirs:
    samples = []
    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for img_path in glob(os.path.join(class_path, "*")):
            samples.append((img_path, global_class_to_idx[class_name]))
    
    dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    dataset.samples = samples
    dataset.targets = [s[1] for s in samples]
    dataset.class_to_idx = global_class_to_idx
    train_datasets.append(dataset)

train_dataset = ConcatDataset(train_datasets)
train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True,
                          num_workers=params.workers, pin_memory=True)

# ==== Load Validation Data ====
val_dataset = datasets.ImageFolder("val.X", transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False,
                        num_workers=params.workers, pin_memory=True)

# ==== Model ====
# model = torchvision.models.resnet50(weights=None)
# model.fc = nn.Linear(model.fc.in_features, params.num_classes)
# model = model.to(device)

# Model parameters
n_fixations = 3  # Number of fixations for the active vision model
radius = 0.6
block_sigma = 0.05
block_max_ord = 2
patch_sigma = 0.05
patch_max_ord = 2
ds_sigma = 0.05
ds_max_ord = 2

# Create model with correct number of classes
model = make_model(
    n_fixations=n_fixations,
    n_classes=100,  # Use full 100 classes
    radius=radius,
    block_sigma=block_sigma,
    block_max_ord=block_max_ord,
    patch_sigma=patch_sigma,
    patch_max_ord=patch_max_ord,
    ds_sigma=ds_sigma,
    ds_max_ord=ds_max_ord
).to(device)

# ==== Loss, Optimizer, LR Scheduler ====
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.SGD(model.parameters(), lr=params.lr,
                      momentum=params.momentum, weight_decay=params.weight_decay)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                         step_size=params.lr_step_size,
                                         gamma=params.lr_gamma)

start_epoch = 0
if params.resume_checkpoint is not None and os.path.isfile(params.resume_checkpoint):
    print(f"Loading checkpoint from {params.resume_checkpoint}")
    checkpoint = torch.load(params.resume_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    print(f"Resuming training from epoch {start_epoch}")

# ==== Train Function ====
def train_one_epoch(epoch):
    model.train()
    total_loss = 0
    correct = 0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        # print("Output range:", outputs.min().item(), outputs.max().item())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

        if i % 100 == 0:
            print(f"Epoch [{epoch}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

    acc = correct / len(train_loader.dataset)
    print(f"Epoch [{epoch}] Training Accuracy: {acc:.4f}, Loss: {total_loss / len(train_loader):.4f}")
    return acc, total_loss / len(train_loader)

# ==== Evaluation Function ====
def evaluate(epoch):
    model.eval()
    correct1 = 0
    correct5 = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()

            _, preds1 = outputs.topk(1, dim=1)
            _, preds5 = outputs.topk(5, dim=1)
            correct1 += (preds1.squeeze() == labels).sum().item()
            correct5 += preds5.eq(labels.view(-1, 1)).sum().item()
            total += labels.size(0)

    top1_acc = correct1 / total
    top5_acc = correct5 / total
    print(f"Epoch [{epoch}] Validation Accuracy: Top-1 {top1_acc:.4f}, Top-5 {top5_acc:.4f}, Loss: {loss_sum / len(val_loader):.4f}")
    return top1_acc, top5_acc, loss_sum / len(val_loader)

# ==== Logging and Checkpointing ====
writer = SummaryWriter(log_dir=f"runs/{params.name}")
os.makedirs(f"checkpoints/{params.name}", exist_ok=True)

# ==== Main Training Loop ====
for epoch in range(start_epoch, params.epochs):
    train_acc, train_loss = train_one_epoch(epoch)
    val_acc1, val_acc5, val_loss = evaluate(epoch)

    writer.add_scalar("Train/Loss", train_loss, epoch)
    writer.add_scalar("Train/Accuracy", train_acc, epoch)
    writer.add_scalar("Val/Top1_Accuracy", val_acc1, epoch)
    writer.add_scalar("Val/Top5_Accuracy", val_acc5, epoch)
    writer.add_scalar("Val/Loss", val_loss, epoch)

    lr_scheduler.step()

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, f"checkpoints/{params.name}/checkpoint_epoch{epoch}.pth")
