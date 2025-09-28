import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from einops import rearrange


class ClfDataset(Dataset):
    def __init__(self, path, train=True):
        super().__init__()

        self.path = path
        self.train = train

        # Load CSV and normalize column names
        self.image_df = pd.read_csv(f"{self.path}/labels.csv")
        self.image_df.columns = self.image_df.columns.str.strip().str.lower()  # lowercase all col names

        # Remove label 0 and shift labels to start from 0
        self.image_df = self.image_df[self.image_df["label"] != 0]
        self.image_df["label"] = self.image_df["label"] - 1

        # Store images and labels
        self.images = list(self.image_df["image"].values)
        self.labels = self.image_df.set_index("image")

        # Define transforms
        self.transform = A.Compose(
            [
                A.Resize(224, 224),
                *(
                    [A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.4)]
                    if train
                    else []
                ),
                A.Crop(x_min=130, y_min=43, x_max=202, y_max=147),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_type = self.labels.loc[image_name]["type"]

        img = Image.open(os.path.join(self.path, image_type, image_name)).convert("RGB")
        img_tensor = self.transform(image=np.array(img))
        label = self.labels.loc[image_name]["label"]

        return img_tensor["image"], torch.tensor(label)


if __name__ == "__main__":

    data = ClfDataset("data/test_data")
    dataloader = DataLoader(data, batch_size=32, shuffle=True)

    X, y = next(iter(dataloader))
    print(X.shape, y.shape)

    # Optional: save transformed samples
    output_samples = False
    if output_samples:
        for idx, img in enumerate(X):
            img_np = img.permute(1, 2, 0).numpy()
            img_min, img_max = img_np.min(), img_np.max()
            img_np_scaled = (img_np - img_min) / (img_max - img_min)
            plt.imsave(f"image/transformed_images/{idx}_scaled.png", img_np_scaled)

    # Optional: patch visualization
    output_patches = False
    if output_patches:
        patch_size = 8
        batch_size, channels, height, width = X.shape

        assert height % patch_size == 0 and width % patch_size == 0, "Height and Width must be divisible by patch_size"

        res = rearrange(X, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size)
        for idx, img in enumerate(res):
            reconstructed_image = rearrange(img, "(h w) (p1 p2 c) -> h w p1 p2 c", h=height//patch_size, w=width//patch_size, p1=patch_size, p2=patch_size)
            fig = plt.figure(figsize=(10, 10))
            grid = ImageGrid(fig, 111, nrows_ncols=(height // patch_size, width // patch_size), axes_pad=0.1)

            for i, ax in enumerate(grid):
                ax.imshow(reconstructed_image.reshape(-1, patch_size, patch_size, 3)[i])
            plt.show()
