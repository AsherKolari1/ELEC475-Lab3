"""Validate both dataset splits and plot eight training nose labels."""
from pathlib import Path
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from .dataset import IMG_SIZE, MEAN, STD, PetNoseDataset


def main():
    # Limit CPU threads: small per-image operations otherwise incur overhead.
    torch.set_num_threads(1)
    preview = None
    for split in ("train", "test"):
        dataset = PetNoseDataset(
            root=os.getenv("ELEC475_DATASET_ROOT", "data/oxford-iiit-pet-noses"),
            split=split,
        )
        workers = int(os.getenv("ELEC475_NUM_WORKERS", "0"))
        loader = DataLoader(
            dataset, batch_size=int(os.getenv("ELEC475_BATCH_SIZE", "64")), shuffle=False,
            num_workers=workers,
            pin_memory=os.getenv("ELEC475_PIN_MEMORY", "0") == "1",
            persistent_workers=workers > 0 and os.getenv("ELEC475_PERSISTENT_WORKERS", "0") == "1",
            prefetch_factor=2 if workers > 0 else None,
        )
        count = 0
        for images, labels, _ in loader:
            if tuple(images.shape[1:]) != (3, IMG_SIZE, IMG_SIZE):
                raise ValueError(f"{split}: invalid image shape {images.shape}")
            valid = torch.isfinite(labels).all(dim=1)
            valid &= ((labels >= 0) & (labels < IMG_SIZE)).all(dim=1)
            if not valid.all():
                indices = (count + (~valid).nonzero().flatten()).tolist()
                raise ValueError(f"{split}: labels outside frame at indices {indices}")
            if split == "train" and preview is None:
                preview = (images[:8].clone(), labels[:8].clone())
            count += len(images)
        print(f"{split}: validated {count:,} samples", flush=True)

    if preview is None or len(preview[0]) != 8:
        raise ValueError("At least eight training samples are required")
    images, labels = preview
    images = (images * torch.tensor(STD).view(3, 1, 1)
              + torch.tensor(MEAN).view(3, 1, 1)).clamp(0, 1)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for index, (ax, image, label) in enumerate(zip(axes.flat, images, labels)):
        ax.imshow(image.permute(1, 2, 0).numpy())
        ax.scatter(*label.tolist(), s=24, c="red")
        ax.set_title(f"Training sample {index}")
        ax.axis("off")
    fig.tight_layout()
    output = Path(os.getenv("ELEC475_OUTPUT_DIR", "plots")) / "dataset_check.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
