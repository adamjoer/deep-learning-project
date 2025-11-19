# from accelerate import Accelerator
from datetime import datetime
from pathlib import Path

import torch

# import wandb
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from unet import UNet
from vdm import VDM

from accelerate import Accelerator
from accelerate.utils import set_seed

BATCH_SIZE = 128
TRAIN_NUM_STEPS = 10_000_000
SAVE_EVERY = 10_000
NUM_WORKERS = 4
LR = 2e-4


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


def save_checkpoint(step: int, vdm: VDM, opt: torch.optim.Optimizer, accelerator: Accelerator):
    checkpoint_file = Path("model.pt")
    tmp_file = checkpoint_file.with_suffix(f".tmp.{datetime.now().isoformat()}.pt")
    if checkpoint_file.exists():
        checkpoint_file.rename(tmp_file)  # Rename old checkpoint to temp file
    checkpoint = {
        "step": step,
        "model": accelerator.get_state_dict(vdm),
        "opt": opt.state_dict(),
    }
    torch.save(checkpoint, checkpoint_file)
    tmp_file.unlink(missing_ok=True)  # Delete temp file

    # wandb.save(str(checkpoint_file))


def main():
    accelerator = Accelerator(split_batches=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # wandb.init(
    #     project="deep-learning-project",
    #     config={
    #         "batch_size": BATCH_SIZE,
    #         "learning_rate": LR,
    #         "num_workers": NUM_WORKERS,
    #         "train_num_steps": TRAIN_NUM_STEPS,
    #         "save_every": SAVE_EVERY,
    #         "device": str(device),
    #     },
    # )

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

    opt = accelerator.prepare(torch.optim.AdamW(vdm.parameters(), lr=LR))
    # opt = torch.optim.AdamW(vdm.parameters(), lr=LR)

    # wandb.watch(vdm, log="all", log_freq=100)

    step = 0
    with tqdm(initial=step, total=TRAIN_NUM_STEPS) as pbar:
        while step < TRAIN_NUM_STEPS:
            (data, label) = next(iter(train_dl))
            opt.zero_grad()
            loss = vdm(data.to(device))
            loss.backward()
            opt.step()
            # wandb.log(
            #     {
            #         "train/loss": loss.item(),
            #         "train/step": step,
            #         "lr": opt.param_groups[0]["lr"],
            #     },
            #     step=step,
            # )

            pbar.set_description(f"loss: {loss.item():.4f}")
            step += 1
            accelerator.wait_for_everyone()
            if step % 100 == 0:
                print(f"[step {step}] loss = {loss.item():.4f}")
            if step % SAVE_EVERY == 0 and step > 0:
                save_checkpoint(step, vdm, opt, accelerator)
            pbar.update()

    # wandb.finish()


if __name__ == "__main__":
    main()
