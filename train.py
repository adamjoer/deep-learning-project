import os
from datetime import datetime
from pathlib import Path

import torch
from accelerate import Accelerator
from ema_pytorch import EMA
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

import wandb
from unet import UNet
from vdm import VDM

BATCH_SIZE = 64
NUM_WORKERS = 4
LR = 1e-4
EPOCHS = 100
VALIDATE_EVERY_EPOCH = 10
EMA_DECAY = 0.9999
EMA_UPDATE_EVERY = 1
EMA_POWER = 3 / 4  # 0.999 at 10k, 0.9997 at 50k, 0.9999 at 200k


def get_cifar10_dataset(root="data", train=False, download=False):
    return CIFAR10(
        root=root,
        train=train,
        transform=transforms.Compose([transforms.ToTensor()]),
        download=download,
    )


def cycle(dl):
    # We don't use itertools.cycle because it caches the entire iterator.
    while True:
        for data in dl:
            yield data


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
                "device": str(device),
            },
        )

    train_set = get_cifar10_dataset(train=True, download=False)
    validation_set = get_cifar10_dataset(train=False, download=False)

    training_dataloader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )
    validation_dataloader = DataLoader(
        validation_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    if accelerator.is_main_process:
        print(f"Training dataset size: {len(train_set)}")
        print(f"Validation dataset size: {len(validation_set)}")
        print(f"Training dataloader size: {len(training_dataloader)}")
        print(f"Validation dataloader size: {len(validation_dataloader)}")
        print(f"Shape: {train_set[0][0].shape}")

    unet = UNet()
    vdm = VDM(unet, image_shape=train_set[0][0].shape, device=device)

    optimizer = optim.AdamW(vdm.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=0.01, eps=1e-8)

    if accelerator.is_main_process:
        wandb.watch(vdm, log="gradients", log_freq=100)

    vdm, optimizer, training_dataloader, validation_dataloader = accelerator.prepare(
        vdm, optimizer, training_dataloader, validation_dataloader
    )

    output_path = Path("./outputs")
    path = output_path / datetime.now().isoformat()
    path.mkdir(exist_ok=True, parents=True)

    checkpoint_file = path / "model.pt"

    ema: EMA | None = None
    if accelerator.is_main_process:
        ema = EMA(
            vdm.to(accelerator.device),
            beta=EMA_DECAY,
            update_every=EMA_UPDATE_EVERY,
            power=EMA_POWER,
        )
        if ema.ema_model and isinstance(ema.ema_model, nn.Module):
            ema.ema_model.eval()

    def save_checkpoint(epoch):
        tmp_file = checkpoint_file.with_suffix(f".tmp.{datetime.now().isoformat()}.pt")
        if checkpoint_file.exists():
            checkpoint_file.rename(tmp_file)  # Rename old checkpoint to temp file
        checkpoint = {
            "step": epoch,
            "model": accelerator.get_state_dict(vdm),
            "opt": optimizer.state_dict(),
            "ema": ema.state_dict() if ema is not None else None,
        }
        torch.save(checkpoint, checkpoint_file)
        tmp_file.unlink(missing_ok=True)  # Delete temp file

        wandb.save(str(checkpoint_file))

    losses: list[float] = []
    validation_losses: list[float] = []
    bpd_list: list[float] = []
    validation_bpd_list: list[float] = []
    bpd_recon_list: list[float] = []
    bpd_klz_list: list[float] = []
    bpd_diff_list: list[float] = []

    for epoch in (progress_bar := tqdm(range(EPOCHS), disable=not accelerator.is_main_process)):
        vdm.train()
        cumulative_loss = 0.0
        cumulative_bpd = 0.0
        cumulative_bpd_recon = 0.0
        cumulative_bpd_klz = 0.0
        cumulative_bpd_diff = 0.0

        for batch in (
            train_progress_bar := tqdm(
                training_dataloader, position=1, leave=False, disable=not accelerator.is_main_process
            )
        ):
            optimizer.zero_grad()

            loss, bpd, bpd_components = vdm(batch[0])
            accelerator.backward(loss)

            accelerator.clip_grad_norm_(vdm.parameters(), 1.0)
            optimizer.step()
            cumulative_loss += loss.item()
            cumulative_bpd += bpd.item()
            cumulative_bpd_recon += bpd_components["bpd_recon"]
            cumulative_bpd_klz += bpd_components["bpd_klz"]
            cumulative_bpd_diff += bpd_components["bpd_diff"]

            if accelerator.is_main_process:
                train_progress_bar.set_description(
                    f"loss: {loss.item():.4f}, bpd: {bpd.item():.4f} "
                    f"(R:{bpd_components['bpd_recon']:.3f} K:{bpd_components['bpd_klz']:.3f} D:{bpd_components['bpd_diff']:.3f})"
                )
                if ema:
                    ema.update()

        accelerator.wait_for_everyone()

        average_loss = cumulative_loss / len(training_dataloader)
        average_bpd = cumulative_bpd / len(training_dataloader)
        average_bpd_recon = cumulative_bpd_recon / len(training_dataloader)
        average_bpd_klz = cumulative_bpd_klz / len(training_dataloader)
        average_bpd_diff = cumulative_bpd_diff / len(training_dataloader)

        losses.append(average_loss)
        bpd_list.append(average_bpd)
        bpd_recon_list.append(average_bpd_recon)
        bpd_klz_list.append(average_bpd_klz)
        bpd_diff_list.append(average_bpd_diff)

        if accelerator.is_main_process:
            wandb.log(
                {
                    "train/loss": average_loss,
                    "train/bpd": average_bpd,
                    "train/bpd_recon": average_bpd_recon,
                    "train/bpd_klz": average_bpd_klz,
                    "train/bpd_diff": average_bpd_diff,
                    "train/epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

            save_checkpoint(epoch)

        # Validation
        average_validation_loss = 0.0
        average_validation_bpd = 0.0
        if epoch % VALIDATE_EVERY_EPOCH == 0:
            vdm.eval()
            cumulative_validation_loss = 0.0
            cumulative_validation_bpd = 0.0

            with torch.no_grad():
                for batch in validation_dataloader:
                    validation_loss, validation_bpd, _ = vdm(batch[0])
                    cumulative_validation_loss += validation_loss.item()
                    cumulative_validation_bpd += validation_bpd.item()

            accelerator.wait_for_everyone()

            average_validation_loss = cumulative_validation_loss / len(validation_dataloader)
            average_validation_bpd = cumulative_validation_bpd / len(validation_dataloader)
            validation_losses.append(average_validation_loss)
            validation_bpd_list.append(average_validation_bpd)

            if accelerator.is_main_process:
                wandb.log(
                    {
                        "validation/loss": average_validation_loss,
                        "validation/bpd": average_validation_bpd,
                        "validation/epoch": epoch,
                    },
                    step=epoch,
                )
        else:
            validation_losses.append(validation_losses[-1] if validation_losses else 0.0)
            validation_bpd_list.append(validation_losses[-1] if validation_losses else 0.0)

        progress_bar.set_description(
            f"loss: {average_loss:.4f}, val_loss: {average_validation_loss:.4f}, "
            f"bpd: {average_bpd:.4f}, val_bpd: {average_validation_bpd:.4f}"
        )

    wandb.finish()

    # Save cumulative losses and BPD to a CSV file
    if accelerator.is_main_process:
        import pandas as pd

        df = pd.DataFrame(
            {
                "epoch": list(range(EPOCHS)),
                "loss": losses,
                "validation_loss": validation_losses,
                "bpd": bpd_list,
                "validation_bpd": validation_bpd_list,
                "bpd_recon": bpd_recon_list,
                "bpd_klz": bpd_klz_list,
                "bpd_diff": bpd_diff_list,
            }
        )
        df.to_csv(path / "losses.csv", index=False)


if __name__ == "__main__":
    main()
