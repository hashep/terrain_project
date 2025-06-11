import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model import get_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_path = "C:/Anaconda/terrain_project/data/images/image.jpg"
    model_path = "C:/Anaconda/terrain_project/model/model_epoch_50.pth"
    num_classes = 9

    original_image = Image.open(image_path).convert("RGB")
    original_size = original_image.size
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    input_tensor = transform(original_image).unsqueeze(0).to(device)

    model = get_model(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        output = model(input_tensor) 
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy() 

    pred_mask_img = Image.fromarray(pred_mask.astype(np.uint8))
    pred_mask_img = pred_mask_img.resize(original_size, resample=Image.NEAREST)

    colors = np.array([
        [0,0,0],       
        [128,0,0],     
        [0,128,0],     
        [128,128,0],  
        [0,0,128],    
        [128,0,128],  
        [0,128,128],   
        [128,128,128], 
        [64,64,64]     
    ])

    pred_mask_color = colors[np.array(pred_mask_img)]

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask_color)
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()
