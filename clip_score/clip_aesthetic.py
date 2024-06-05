import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import clip
import os

from PIL import Image
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            # nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

def calculate_clip_aesthetic(img_path):
    model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

    s = torch.load(
        "D:\\workspace\\PythonProject\\criteria\\checkpoint\\clip_aesthetic\\ava+logos-l14-linearMSE.pth")  # load the model you trained previously or the model available in this repo

    model.load_state_dict(s)

    model.to("cuda")
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model2, preprocess = clip.load("ViT-L/14", device=device)  # RN50x64

    if os.path.isdir(img_path):
        tensor_list= []
        for image_path in os.listdir(img_path):
            pil_image = Image.open(os.path.join(img_path, image_path))
            image = preprocess(pil_image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model2.encode_image(image)
            im_emb_arr = normalized(image_features.cpu().detach().numpy())
            prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))
            tensor_list.append(prediction)
        stacked_tensors = torch.stack(tensor_list)
        prediction = torch.mean(stacked_tensors)

    elif os.path.isfile(img_path):
        pil_image = Image.open(img_path)
        image = preprocess(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model2.encode_image(image)
        im_emb_arr = normalized(image_features.cpu().detach().numpy())
        prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))

    print("Aesthetic score predicted by the model:")
    print(prediction)
    return prediction

if __name__ == '__main__':
    image_path = "D:\\workspace\\PythonProject\\criteria\\test_img\\test.png"
    calculate_clip_aesthetic(image_path)