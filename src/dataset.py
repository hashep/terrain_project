import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch

# 마스크 픽셀값 → 연속 클래스 인덱스 매핑 (본인 데이터에 맞게 수정하세요)
PIXEL2LABEL = {
    0: 0,
    1: 1,
    2: 2,
    4: 3,
    8: 4,
    10: 5,
    15: 6,
    19: 7,
    22: 8
}

class DroneDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transforms=None, mask_transforms=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, self.images[idx])).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, self.masks[idx])).convert("L")

        if self.image_transforms:
            image = self.image_transforms(image)
        if self.mask_transforms:
            mask = self.mask_transforms(mask)

        mask_np = np.array(mask)
        mask_mapped = np.zeros_like(mask_np, dtype=np.int64)
        for k, v in PIXEL2LABEL.items():
            mask_mapped[mask_np == k] = v

        mask_tensor = torch.as_tensor(mask_mapped, dtype=torch.long)
        return image, mask_tensor
