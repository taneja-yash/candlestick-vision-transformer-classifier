from model import ViT
from data import ClfDataset
from torch.utils.data import DataLoader
import torch
from matplotlib import pyplot as plt

candles = {
    0:"Doji",
    1:"Bullish Engulfing",
    2:"Bearish Engulfing",
    3:"Morning Star",
    4:"Evening Star"
}

model = ViT()

model.load_state_dict(torch.load("checkpoints/10_model.pt", weights_only=True, map_location=torch.device('cpu')))
model.eval()

data = ClfDataset("data/test_data", train=False)
dataloader = DataLoader(
    data, batch_size=9, shuffle=True
)

sample = next(iter(dataloader))
X = sample[0]
print(X.shape)
preds = model(X)

softmax = torch.nn.Softmax(dim=1)
print(softmax(preds))

argmax = torch.argmax(softmax(preds), dim=-1)
print(argmax)
print(sample[1])

loss_fn = torch.nn.CrossEntropyLoss()
print(loss_fn(preds, sample[1]))

fig, ax = plt.subplots(3,3)
axs = ax.flatten()
for act, pred, img, ax in zip(sample[1], argmax, X, axs):
    img_np = img.permute(1,2,0).numpy()
    img_min, img_max = img_np.min(), img_np.max()
    img_np_scaled = (img_np - img_min) / (img_max - img_min)
    ax.imshow(img_np_scaled)
    ax.set_title(f"Pred: {candles[pred.item()]}\nAct: {candles[act.item()]}", fontsize=10)
fig.tight_layout()
plt.savefig('results.png')