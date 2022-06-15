import shutil

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib import gridspec
from torch import nn, optim
from torchvision.transforms import transforms
from torch.utils.data import random_split
import torch.nn.functional as F
from torch.utils.data import RandomSampler

from lidarNet.dataset import calculate_input_dim, LidarEnlarge, LidarNetDataset, calculate_unet_output_dim
from lidarNet.utils.geo_utils import visualise_height_data, create_raster_from_transform


def crop_masks(masks, dims):
    # dims smaller than masks
    if masks.shape[-2] < dims[0] or masks.shape[-1] < dims[1]:
        raise ValueError("Dims must be smaller than masks")
    is_batch = False
    if len(masks.shape) == 4:
        # batch, channels, y, x
        is_batch = True
    h_old, w_old = masks.shape[-2:]
    h_new, w_new = dims
    h_diff = h_old - h_new
    w_diff = w_old - w_new
    top, bottom = h_diff // 2, h_diff - h_diff // 2
    left, right = w_diff // 2, w_diff - w_diff // 2
    if not is_batch:
        return masks[:, top: h_old - bottom, left: w_old - right]
    return masks[:, :, top: h_old - bottom, left: w_old - right]


RUN_TRAINING = True
UNET_LAYERS = 4
IMAGE_DIMS = 590
DOWNSCALE_FACTOR = 2
BATCH_SIZE = 5
unet_input_dim = calculate_input_dim(int(IMAGE_DIMS // DOWNSCALE_FACTOR), UNET_LAYERS) * 2

unet_output_upscale = transforms.Resize(IMAGE_DIMS)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Resize(int(IMAGE_DIMS // DOWNSCALE_FACTOR)),
        LidarEnlarge((unet_input_dim, unet_input_dim))
    ]
)

train_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Resize(int(IMAGE_DIMS // DOWNSCALE_FACTOR)),
        transforms.ColorJitter(brightness=0.3, contrast=0.7, saturation=0.7, hue=0.05),
        LidarEnlarge((unet_input_dim, unet_input_dim))
    ]
)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_layers = []
        decoder_layers = []

        # channels = [3, 64, 128, 256, 512, 1024]
        channels = [3]
        for i in range(UNET_LAYERS + 1):
            channels.append(2 ** (i + 6))
        for prev_c, c in zip(channels[:-1], channels[1:-1]):
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(prev_c, c, 3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(c, c, 3),
                    nn.ReLU(inplace=True),
                )
            )
            encoder_layers.append(
                nn.MaxPool2d(kernel_size=2)
            )

        self.bottom_layer = nn.Sequential(
            nn.Conv2d(channels[-2], channels[-1], 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[-1], channels[-1], 3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[-1], channels[-2], kernel_size=2, stride=2)
        )

        for i in reversed(range(3, len(channels))):
            layer = nn.Sequential(
                nn.Conv2d(channels[i], channels[i - 1], 3),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels[i - 1], channels[i - 1], 3),
                nn.ReLU(inplace=True),
            )
            layer.append(nn.ConvTranspose2d(channels[i - 1], channels[i - 2], kernel_size=2, stride=2))
            decoder_layers.append(layer)
        decoder_layers.append(nn.Sequential(
            nn.Conv2d(channels[2], channels[1], 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[1], channels[1], 3),
            nn.ReLU(inplace=True),
        ))

        self.final_layer = nn.Sequential(
            nn.Conv2d(channels[1], 1, 1)
        )
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

        self.downscale = nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=2, padding=1)
        self.upscale = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)

    def encode(self, x):
        xs = []
        layers = self.encoder.children()
        for convs in layers:
            maxpool = next(layers)
            x = convs(x)
            xs.append(x)
            x = maxpool(x)
        return x, xs

    @staticmethod
    def concat_layer(z, x):
        # z is smaller one, x is larger
        x_crop = crop_masks(x, z.shape[-2:])
        return torch.cat((z, x_crop), dim=1)

    def decode(self, z, xs):
        z = self.bottom_layer(z)
        for i, l in enumerate(self.decoder.children()):
            z = self.concat_layer(z, xs[-(i + 1)])
            z = l(z)
        return self.final_layer(z)

    def forward(self, x):
        x = self.downscale(x)
        z, xs = self.encode(x)
        m = self.decode(z, xs)
        m = self.upscale(m)
        return m


train_test_split = 0.9
dataset = LidarNetDataset(
    "./lidarNet/data/london/rgb",
    "./lidarNet/data/london/lidar",
    transform=transform
)
split_index = int(len(dataset) * train_test_split)
train_dataset, test_dataset = random_split(dataset, [split_index, len(dataset) - split_index])
train_dataset.dataset.transform = train_transforms
test_size = len(test_dataset)
test_dataset, val_dataset = random_split(test_dataset, [test_size // 2, test_size - test_size // 2])
print("CUDA working:", torch.cuda.is_available())
print("Data shape:", train_dataset[0][0].shape, "Mask shape:", train_dataset[0][1].shape)
print("Training data:", len(train_dataset), "Testing data:", len(test_dataset), "Validation data:", len(val_dataset))

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)


EPOCHS = 1
criterion = nn.L1Loss()

loss_history = []
val_loss_history = []

device = torch.device("cuda:0")


def make_model(config):
    mdl = UNet().to(device)
    optimizer = optim.RMSprop(mdl.parameters(), lr=config["learning_rate"], weight_decay=1e-8, momentum=config["momentum"])
    return mdl, optimizer


def val_loss(mdl, val_loader):
    running_loss = 0.0
    mdl.eval()
    with torch.no_grad():
        for i, d in enumerate(val_loader):
            inputs, masks = d
            inputs = inputs.to(device)
            masks = masks.to(device)

            outputs = mdl(inputs).squeeze(dim=1)
            masks = crop_masks(masks, outputs.shape[1:3])
            loss = criterion(outputs, masks)

            running_loss += loss.item()
    return running_loss


def dice_loss(inputs, targets, eps=1):
    assert inputs.shape == targets.shape
    inputs = inputs.reshape(-1)
    targets = targets.reshape(-1)
    intersection = (inputs * targets).sum()
    dice = (2 * intersection + eps) / (inputs.sum() + targets.sum() + eps)
    return 1 - dice


def train(mdl, opt, test_loader):
    mdl.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, (inputs, masks) in enumerate(test_loader):
            inputs = inputs.to(device)
            masks = masks.to(device)

            opt.zero_grad()
            outputs = mdl(inputs).squeeze(dim=1)
            masks = crop_masks(masks, outputs.shape[1:3])
            loss = criterion(outputs, masks)
            loss.backward()
            opt.step()

            running_loss += loss.item()
            if i % 10 == 9:    # print every 2000 mini-batches
                v_loss = val_loss(mdl, valloader)
                mdl.train()
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f} val_loss: {v_loss}')
                loss_history.append(loss)
                val_loss_history.append(v_loss)
                # wandb.log({"loss": running_loss, "val_loss": v_loss})
                running_loss = 0.0


if RUN_TRAINING:
    mdl = UNet().to(device)
    train(mdl, optim.RMSprop(mdl.parameters(), lr=4e-5, weight_decay=1e-8, momentum=0.93), trainloader)
    torch.save(mdl.state_dict(), "./model.pt")
else:
    mdl = UNet().to(device)
    mdl.load_state_dict(torch.load("./model.pt"))
mdl.eval()

# plt.plot([x.item() for x in loss_history][10:])
# plt.plot(val_loss_history[10:])


def visualise_test(mdl):
    mdl.eval()
    n_col = 6
    n_row = 1

    r = RandomSampler(test_dataset, num_samples=n_col)

    n_col //= 2
    n_row *= 2
    fig = plt.figure(figsize=(n_row + 1, n_col + 2))
    axs = []
    gs = gridspec.GridSpec(n_col, n_row, wspace=0.0, hspace=0.0,
       top=1. - 0.5 / (n_col + 1), bottom=0.5 / (n_col + 1),
       left=0.5 / (n_row + 1), right=1 - 0.5 / (n_row + 1)
    )

    for i in range(n_col):
        for j in range(n_row):
            ax = plt.subplot(gs[i, j])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            axs.append(ax)

    for i, d_idx in enumerate(r):
        test_in, test_gt = test_dataset[d_idx]
        test_in = test_in.to(device)

        with torch.no_grad():
            outputs = mdl(test_in.unsqueeze(dim=0)).squeeze(dim=0)
        test_gt = crop_masks(np.expand_dims(test_gt, axis=0), outputs.shape[-2:])
        test_net = outputs[0].cpu()
        print(test_net.shape)

        test_in = crop_masks(test_in, (325, 325))
        test_in = transforms.Resize((650, 650))(test_in)
        test_in = crop_masks(test_in, outputs.shape[-2:])

        test_in = test_in.cpu()
        test_in_clone = test_in.clone()
        test_in[0][test_gt == 0.0] += 0.3
        test_in[0][test_in[0] > 1.0] = 1.0
        test_in_clone[0][test_net < 0.5] += 0.3
        test_in_clone[0][test_in_clone[0] > 1.0] = 1.0
        visible = test_in.permute(1, 2, 0).detach().numpy()
        height_preds = test_net.detach().numpy()
        visualise_height_data(visible, height_preds)
        axs[i].imshow(visible)
        # axs[i, 0].imshow(test_gt)
        # axs[(i * 2) + 1].imshow(test_net)


def test_from_raster(col: int, row: int, rgb_dir: str, lidar_dir: str, out_dir: str):
    rgb_raster_fp = f"{rgb_dir}/col{col}_row{row}_rgb.tif"
    lidar_raster_fp = f"{lidar_dir}/col{col}_row{row}_ndsm.tif"
    rgb_raster = rasterio.open(rgb_raster_fp)

    mdl.eval()
    test_in = np.transpose(rgb_raster.read((3, 2, 1)), axes=(1, 2, 0))
    img = np.transpose(np.copy(test_in), axes=(2, 0, 1))
    test_in = transform(test_in)
    test_in = test_in.to(device)

    with torch.no_grad():
        outputs = mdl(test_in.unsqueeze(dim=0)).squeeze(dim=0)
    height_preds = outputs[0].cpu()
    height_preds = LidarEnlarge(img.shape[-2:])(torch.unsqueeze(height_preds, dim=0))

    create_raster_from_transform(
        rgb_raster.transform,
        rgb_raster.crs,
        height_preds.detach().numpy(),
        f"./{out_dir}/model_preds_1_epoch_downscaling.tif",
        height_preds.shape[-2:],
        channels=1
    )

    shutil.copy(rgb_raster_fp, f"{out_dir}/rgb.tif")
    shutil.copy(lidar_raster_fp, f"{out_dir}/gt.tif")


test_from_raster(55, 35, "./lidarNet/data/london/rgb", "./lidarNet/data/london/lidar", "./testing")
