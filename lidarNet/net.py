import torch
from torch import optim
from torch.utils.data import random_split
from torchvision.transforms import transforms

from lidarNet.dataset import calculate_input_dim, LidarNetDataset
from lidarNet.loss_funcs import val_loss, l1_loss_custom
from lidarNet.models.unet import UNet
from lidarNet.models.unet_res import UNetRes
from lidarNet.utils.geo_utils import LatLong
from lidarNet.utils.ml_utils import run_from_coords
from lidarNet.visualise import vis_row_2

RUN_TRAINING = False
CONTINUE_TRAINING = "./model_70.pt"
UNET_LAYERS = 4
IMAGE_DIMS = 590
DOWNSCALE_FACTOR = 2
RESIZE_DIM = 512

BATCH_SIZE = 3
unet_input_dim = calculate_input_dim(int(IMAGE_DIMS // DOWNSCALE_FACTOR), UNET_LAYERS) * 2
unet_input_dim = IMAGE_DIMS

unet_output_upscale = transforms.Resize(IMAGE_DIMS)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(RESIZE_DIM),
        # LidarEnlarge((unet_input_dim, unet_input_dim))
    ]
)

train_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(RESIZE_DIM),
        transforms.ColorJitter(brightness=0.3, contrast=0.7, saturation=0.7, hue=0.05),
        # LidarEnlarge((unet_input_dim, unet_input_dim))
    ]
)

target_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(RESIZE_DIM // 2)
    ]
)

train_test_split = 0.9
dataset = LidarNetDataset(
    "./lidarNet/data/london/rgb",
    "./lidarNet/data/london/lidar",
    transform=transform,
    target_transform=target_transforms
)
split_index = int(len(dataset) * train_test_split)
train_dataset, test_dataset = random_split(
    dataset,
    [split_index, len(dataset) - split_index],
    generator=torch.Generator().manual_seed(42)
)
train_dataset.dataset.transform = train_transforms
test_size = len(test_dataset)
test_dataset, val_dataset = random_split(
    test_dataset,
    [test_size // 2, test_size - test_size // 2],
    generator=torch.Generator().manual_seed(42)
)
print("CUDA working:", torch.cuda.is_available())
print("Data shape:", train_dataset[0][0].shape, "Mask shape:", train_dataset[0][1].shape)
print("Training data:", len(train_dataset), "Testing data:", len(test_dataset), "Validation data:", len(val_dataset))

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

EPOCHS = 10
criterion = l1_loss_custom

loss_history = []
val_loss_history = []

device = torch.device("cuda:0")


def make_model(config):
    mdl = UNet(UNET_LAYERS).to(device)
    optimizer = optim.RMSprop(mdl.parameters(), lr=config["learning_rate"], weight_decay=1e-8,
                              momentum=config["momentum"])
    return mdl, optimizer


def train(mdl, opt, test_loader):
    mdl.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, (inputs, masks) in enumerate(test_loader):
            inputs = inputs.to(device)
            masks = masks.to(device).squeeze(dim=1)

            opt.zero_grad()
            outputs = mdl(inputs).squeeze(dim=1)
            # print(inputs.shape, outputs.shape, masks.shape)
            # masks = crop_masks(masks, outputs.shape[1:3])
            loss = criterion(outputs, masks)
            loss.backward()
            opt.step()

            # fig, axs = plt.subplots(1, 3)
            # axs[0].imshow(inputs[0].cpu().permute(1, 2, 0))
            # axs[1].imshow(masks[0].cpu().detach().numpy())
            # axs[2].imshow(outputs[0].cpu().detach().numpy())
            # plt.show()

            running_loss += loss.item()
            if i % 10 == 9:  # print every 2000 mini-batches
                v_loss = val_loss(mdl, valloader, device, criterion)
                mdl.train()
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f} val_loss: {v_loss:.3f}')
                loss_history.append(loss)
                val_loss_history.append(v_loss)
                # wandb.log({"loss": running_loss, "val_loss": v_loss})
                running_loss = 0.0


mdl_cls = UNetRes
if RUN_TRAINING:
    mdl = mdl_cls().to(device)
    if CONTINUE_TRAINING:
        mdl.load_state_dict(torch.load(CONTINUE_TRAINING))
    train(mdl, optim.Adam(mdl.parameters(), lr=1e-4, weight_decay=1e-8), trainloader)
    torch.save(mdl.state_dict(), "./model_80.pt")
else:
    mdl = mdl_cls().to(device)
    mdl.load_state_dict(torch.load("./model_70.pt"))
mdl.eval()

# countryside_dataset = LidarNetDataset(
#     "./lidarNet/data/salisbury/rgb",
#     "./lidarNet/data/salisbury/lidar",
#     transform=transform,
#     target_transform=target_transforms
# )
#
# countryside_loader = torch.utils.data.DataLoader(countryside_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
# eval_loader = countryside_loader
#
# testloss_mae = val_loss(mdl, eval_loader, device, l1_loss_custom)
# print("Test MAE:", len(eval_loader), testloss_mae, testloss_mae / len(eval_loader))
#
# testloss_rmse = val_loss(mdl, eval_loader, device, rmse_loss_custom)
# print("Test RMS:", len(eval_loader), testloss_rmse, testloss_rmse / len(eval_loader))

test_name = "ram"
run_from_coords(LatLong.from_reverse(-0.1776889, 51.5024446), mdl, transform, device, "./testing", test_name)
vis_row_2(f"./testing/rgb_{test_name}.tif", f"./testing/heights_{test_name}.tif")

# row, col = 26, 40
# test_from_raster(col, row, "./lidarNet/data/salisbury/rgb", "./lidarNet/data/salisbury/lidar", "./testing", mdl, transform, device)
# vis_row_3(f"./testing/rgb_{col},{row}.tif", f"./testing/gt_{col},{row}.tif", f"./testing/preds_{col},{row}.tif")
