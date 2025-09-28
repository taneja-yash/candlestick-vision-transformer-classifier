from model import ViT
from data import ClfDataset
import torch
from torch import nn, optim, save
from torch.utils.data import DataLoader, random_split, default_collate
from colorama import Fore
from torchvision.transforms import v2
from torchinfo import summary

if __name__ == "__main__":
    torch.manual_seed(42)
    train_val_data = ClfDataset("data/train_data", train=True)
    test_data = ClfDataset("data/test_data", train=False)

    train_size = int(0.7 * len(train_val_data))
    val_size = len(train_val_data) - train_size
    train_data, val_data = random_split(train_val_data, [train_size, val_size])

cutmix = v2.CutMix(num_classes=5)
mixup = v2.MixUp(num_classes=5)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup, v2.Identity()], p=[0.25, 0.25, 0.5])

def collate_fn(batch):
    return cutmix_or_mixup(*default_collate(batch))

train_dataset = DataLoader(
    train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)

val_dataset = DataLoader(
    val_data, batch_size=16, shuffle=False)

test_dataset = DataLoader(
    test_data, batch_size=16, shuffle=False)

model = ViT()
summary(model, (1, 3, 104, 72))

epochs = 30
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, len(train_dataset)*30, T_mult=2)

train_batches = len(train_dataset)
print(Fore.LIGHTYELLOW_EX + "Starting Training" + Fore.RESET)

for epoch in range(epochs):
    model.train()

    epoch_loss = 0.0
    for batch_idx, batch in enumerate(train_dataset):
        X, y = batch
        yhat = model(X)
        loss = loss_fn(yhat, y)
        epoch_loss += loss.item()

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        progress = (batch_idx + 1) / train_batches
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f"\rEpoch {epoch+1}/{epochs} |{bar}| {batch_idx+1}/{train_batches}", end="")

    print(f" - Train Loss: {epoch_loss/train_batches:.4f}", end="")
    model.eval()
    with torch.no_grad():
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(val_dataset):
            X, y = batch
            yhat = model(X)
            loss = loss_fn(yhat, y)
            epoch_loss += loss.item()

        print(f" - Val Loss: {epoch_loss/len(val_dataset):.4f}", end="")

        for batch_idx, batch in enumerate(test_dataset):
            X, y = batch
            yhat = model(X)
            loss = loss_fn(yhat, y)
            epoch_loss += loss.item()

        print(f" - Test Loss: {epoch_loss/len(test_dataset):.4f}")

    if epoch % 5 == 0:
        save(model.state_dict(), f"checkpoints/{epoch}_model.pt")

save(model.state_dict(), f"checkpoints/final_model.pt")