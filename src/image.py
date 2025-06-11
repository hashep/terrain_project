from PIL import Image
import numpy as np

mask_path = "C:/Anaconda/terrain_project/data/masks/000.png"
mask = Image.open(mask_path).convert("RGB")
colors = np.unique(np.array(mask).reshape(-1, 3), axis=0)
print(colors)  