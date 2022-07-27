# We use torchvision for faster image pre-processing. The transforms are implemented as nn.Module,
# so we jit it to be faster.

import torch
from PIL import Image
from torchvision.transforms import (
    RandomAffine,
    RandomPerspective,
    RandomAutocontrast,
    RandomEqualize,
)
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode

class Transform(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            Resize([image_size], interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            RandomAffine(
                degrees=15,
                translate=(0.1, 0.1),
                scale=(0.8, 1.2),
                shear=(-15, 15, -15, 15),
                interpolation=InterpolationMode.BILINEAR,
                fill=127,
            ),
            RandomPerspective(
                distortion_scale=0.3,
                p=0.3,
                interpolation=InterpolationMode.BILINEAR,
                fill=127,
            ),
            RandomAutocontrast(p=0.3),
            RandomEqualize(p=0.3),
            ConvertImageDtype(torch.float),
            Normalize(mean, std),
        )

    def forward(self, x) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
        return x


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "return_loss": True,
    }
