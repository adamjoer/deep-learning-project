# from accelerate import Accelerator
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from unet import UNet
from vdm import VDM

BATCH_SIZE = 128
TRAIN_NUM_STEPS = 10_000_000
NUM_WORKERS = 4
LR = 2e-4
EPOCHS = 1


def get_cifar10_dataset(root="data", train=False, download=False):
    return CIFAR10(
        root=root,
        train=train,
        transform=transforms.Compose([transforms.ToTensor()]),
        download=download,
    )


def training_step(vdm: VDM, x0: torch.Tensor):
    x0 = x0.to(next(vdm.parameters()).device)
    loss = vdm(x0)
    return loss


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print("Using device:", device)

    # accelerator = Accelerator(split_batches=True)
    train_set = get_cifar10_dataset(train=True, download=False)
    validation_set = get_cifar10_dataset(train=False, download=False)

    train_dl = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )

    unet = UNet()
    vdm = VDM(unet, image_shape=train_set[0][0].shape, device=device).to(device)

    opt = torch.optim.AdamW(vdm.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        for step, (data, label) in enumerate(train_dl):
            opt.zero_grad()
            loss = vdm(data.to(device))
            loss.backward()
            opt.step()

            if step % 100 == 0:
                print(f"[step {step}] loss = {loss.item():.4f}")

            if step == 500:  # stop early for testing
                break


if __name__ == "__main__":
    main()
