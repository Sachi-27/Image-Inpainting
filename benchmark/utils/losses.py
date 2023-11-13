import torch.nn as nn
import numpy as np

def psnr(img1, img2):
    mse = nn.MSELoss()(img1, img2)
    if mse == 0:
        print("MSE is 0")
        return 100
    # max pixel value
    PIXEL_MAX = np.max(img1.numpy())
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))