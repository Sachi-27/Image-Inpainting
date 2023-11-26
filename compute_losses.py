# Inpainting using classical methods
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
from utils.losses import mse, l1, psnr, fid, lpips, tv

def get_scores(dataset_path, loss_fns=[nn.MSELoss()], img_sz=(256, 256)):
    
    # list of scores
    scores = []

    # list of original and inpainted images
    orig_imgs = []
    out_imgs = []

    # number of images in the dataset
    num_imgs = len(os.listdir(dataset_path))//3

    for i in range(num_imgs):
        # Full path to the orig image
        full_orig = os.path.join(dataset_path, f"{i}_orig.png")
        # Full path to the inpainted image
        full_inpainted = os.path.join(dataset_path, f"{i}_recon.png")
        
        try:
            # Original image
            orig_img = cv2.imread(full_orig)
            orig_img = cv2.resize(orig_img, img_sz)
            orig_imgs.append(orig_img)
            # Inpainted image
            inpainted_img = cv2.imread(full_inpainted)
            inpainted_img = cv2.resize(inpainted_img, img_sz)
            out_imgs.append(inpainted_img)
        except:
            print("Error in resizing image: ", full_orig, " or ", full_inpainted)
            continue
        
        
    # Calculate the score
    for loss_fn in loss_fns:
        score = loss_fn(orig_imgs, out_imgs, img_sz)
        scores.append(score)

    return scores


# Inpainting using Telea's algorithm or the fast marching method
if __name__ == "__main__":
    # path to dataset
    dataset_path = ["./Autoencoder/animals_test_out", "./GLCIC_holes/animals_test_out", "./Jiahui_CA_Statistical/animals_test_mosaic_out", "./Jiahui_CA_Statistical/animals_test_hole_out", "./Jiahui_CA_Statistical/animals_test_imgnet_out"]
    save_path = ["scores_Autoencoder_256.txt", "scores_GLCIC_hole_256.txt", "scores_Jiahui_CA_mosaic_256.txt", "scores_Jiahui_CA_hole_256.txt", "scores_Jiahui_CA_imgnet_256.txt"]
    ind = 0

    # get scores
    scores = get_scores(dataset_path[ind], loss_fns=[mse, l1, psnr, fid, lpips, tv], img_sz=(256, 256))

    # get the average score
    out_file = open(save_path[ind], "w")
    out_file.write(f"Average MSE score for GLCIC_hole's algorithm: {scores[0]}\n")
    out_file.write(f"Average L1 score for GLCIC_hole's algorithm: {scores[1]}\n")
    out_file.write(f"Average PSNR score for GLCIC_hole's algorithm: {scores[2]}\n")
    out_file.write(f"Average FID score for GLCIC_hole's algorithm: {scores[3]}\n")
    out_file.write(f"Average LPIPS score for GLCIC_hole's algorithm: {scores[4]}\n")
    out_file.write(f"Average TV score for GLCIC_hole's algorithm: {scores[5]}\n")

