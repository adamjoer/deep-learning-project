# from accelerate import Accelerator
from datetime import datetime
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

import wandb
from unet import UNet
from vdm import VDM

BATCH_SIZE = 64
# TRAIN_NUM_STEPS = 10_000_000
SAVE_EVERY = 1
NUM_WORKERS = 4
LR = 1e-4
EPOCHS = 10


def get_cifar10_dataset(root="data", train=False, download=False):
    return CIFAR10(
        root=root,
        train=train,
        transform=transforms.Compose([transforms.ToTensor()]),
        download=download,
    )


# def training_step(vdm: VDM, x0: torch.Tensor):
#     x0 = x0.to(next(vdm.parameters()).device)
#     loss = vdm(x0)
#     return loss


def cycle(dl):
    # We don't use itertools.cycle because it caches the entire iterator.
    while True:
        for data in dl:
            yield data


def main():
    accelerator = Accelerator(split_batches=True)

    device = accelerator.device
    print("Using device:", device)

    if accelerator.is_main_process:
        wandb.init(
            project="deep-learning-project",
            config={
                "batch_size": BATCH_SIZE,
                "learning_rate": LR,
                "num_workers": NUM_WORKERS,
                "epochs": EPOCHS,
                "save_every": SAVE_EVERY,
                "device": str(device),
            },
        )

    # accelerator = Accelerator(split_batches=True)
    train_set = get_cifar10_dataset(train=True, download=False)
    validation_set = get_cifar10_dataset(train=False, download=False)

    training_dataloader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )

    if accelerator.is_main_process:
        print(f"Loaded batches of size: {training_dataloader.batch_size}")
        print(f"Number of batches: {len(training_dataloader)}")
        print(f"Shape: {train_set[0][0].shape}")

    unet = UNet()
    vdm = VDM(unet, image_shape=train_set[0][0].shape, device=device)

    optimizer = torch.optim.AdamW(vdm.parameters(), lr=LR)

    training_dataloader = training_dataloader

    # opt = torch.optim.AdamW(vdm.parameters(), lr=LR)

    if accelerator.is_main_process:
        wandb.watch(vdm, log="all", log_freq=100)

    vdm, optimizer, training_dataloader = accelerator.prepare(vdm, optimizer, training_dataloader)

    # step = 0

    checkpoint_file = Path("model.pt")

    def save_checkpoint(epoch):
        tmp_file = checkpoint_file.with_suffix(f".tmp.{datetime.now().isoformat()}.pt")
        if checkpoint_file.exists():
            checkpoint_file.rename(tmp_file)  # Rename old checkpoint to temp file
        checkpoint = {
            "step": epoch,
            "model": accelerator.get_state_dict(vdm),
            "opt": optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_file)
        tmp_file.unlink(missing_ok=True)  # Delete temp file

        wandb.save(str(checkpoint_file))

    # train_dl_iter = cycle(training_dataloader)
    # with tqdm(initial=step, total=TRAIN_NUM_STEPS, disable=not accelerator.is_main_process) as pbar:
    #     while step < TRAIN_NUM_STEPS:
    #         (data, label) = next(train_dl_iter)
    #         optimizer.zero_grad()
    #         loss = vdm(data)
    #         accelerator.backward(loss)
    #         optimizer.step()
    #         wandb.log(
    #             {
    #                 "train/loss": loss.item(),
    #                 "train/step": step,
    #                 "lr": optimizer.param_groups[0]["lr"],
    #             },
    #             step=step,
    #         )

    #         pbar.set_description(f"loss: {loss.item():.4f}")
    #         step += 1
    #         accelerator.wait_for_everyone()
    #         if accelerator.is_main_process:
    #             if step % 100 == 0:
    #                 print(f"[step {step}] loss = {loss.item():.4f}")
    #             if step % SAVE_EVERY == 0 and step > 0:
    #                 save_checkpoint()
    #         pbar.update()

    cumulative_losses: list[float] = []

    vdm.train()
    for epoch in (progress_bar := tqdm(range(EPOCHS), disable=not accelerator.is_main_process)):
        cumulative_loss = 0.0
        for batch in training_dataloader:
            accelerator.clip_grad_norm_(vdm.parameters(), 1.0)

            optimizer.zero_grad()

            loss = vdm(batch[0])
            accelerator.backward(loss)

            optimizer.step()
            cumulative_loss += loss.item()

        accelerator.wait_for_everyone()

        cumulative_loss = cumulative_loss / len(training_dataloader)
        cumulative_losses.append(cumulative_loss)

        progress_bar.set_description(f"loss: {cumulative_loss:.4f}")
        if accelerator.is_main_process:
            wandb.log(
                {
                    "train/loss": cumulative_loss,
                    "train/epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

        # TODO: Validate on validation set

        if accelerator.is_main_process:
            if epoch % SAVE_EVERY == 0 and epoch > 0:
                save_checkpoint(epoch)

    wandb.finish()

    # Save cumulative losses to a CSV file
    if accelerator.is_main_process:
        import pandas as pd

        df = pd.DataFrame({"epoch": list(range(EPOCHS)), "cumulative_loss": cumulative_losses})
        df.to_csv("cumulative_losses.csv", index=False)


if __name__ == "__main__":
    main()
