# Inpainting using classical methods
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
from utils.losses import psnr, fid

# Inpainting using Navier-Stokes algorithm
# https://docs.opencv.org/3.4/df/d3d/tutorial_py_inpainting.html
# https://docs.opencv.org/3.4/d7/d8b/group__photo__inpaint.html#gaedd30dfa0214fec4c88138b51d678085


def get_masked(orig_img, img_sz=(256, 256)):
    img = orig_img.copy()
    img[img_sz[0]//4:3*img_sz[0]//4, img_sz[1]//4:3*img_sz[1]//4, :] = 0
    
    mask = np.zeros(img_sz, dtype=np.uint8)
    mask[img_sz[0]//4:3*img_sz[0]//4, img_sz[1]//4:3*img_sz[1]//4] = 255
    
    return img, mask


def get_scores(dataset_path, save_path, img_sz, loss_fns=[nn.MSELoss()], method="telea", save=False):
    
    # extentions of the images
    extentions = ["png", "jpg", "jpeg", "bmp"]
    # list of scores
    scores = [[] for _ in range(len(loss_fns))]
    # counter for number of images
    counter = 0

    for entry in os.listdir(dataset_path):
        # Full path to the image
        full = os.path.join(dataset_path, entry)
        # Name of the image
        last = entry.split(".")[0]
        ext = entry.split(".")[-1]

        # Check if the file is an image
        if ext not in extentions:
            continue
        
        try:
            # Original image
            orig_img = cv2.imread(full)
            # Resize the image
            orig_img = cv2.resize(orig_img, img_sz)
        except:
            print("Error in resizing image: ", full)
            continue
        
        if save:
            cv2.imwrite(save_path + "/" + last + ".png", orig_img)
        
        # Masked image and mask
        img, mask = get_masked(orig_img, img_sz)
        if save:
            cv2.imwrite(save_path + "/" + last + "_input.png", img)
            cv2.imwrite(save_path + "/" + last + "_mask.png", mask)

        # Inpaint using Navier-Stokes algorithm
        if method == "ns":
            out = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
            if save:
                cv2.imwrite(save_path + "/" + last + "_ns.png", out)
        # Inpaint using Telea's algorithm
        else:
            out = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
            if save:
                cv2.imwrite(save_path + "/" + last + "_telea.png", out)

        # Calculate the score
        for i, loss_fn in enumerate(loss_fns):
            score = loss_fn(torch.from_numpy(orig_img).float(), torch.from_numpy(out).float())
            scores[i].append(score)

        counter += 1
        if counter % 100 == 0:
            print("Done with ", counter, " images for method: ", method, " and loss: ", loss_fn.__class__.__name__)

    return scores


# Inpainting using Telea's algorithm or the fast marching method
if __name__ == "__main__":
    # path to dataset
    dataset_path = "./benchmark/animals"
    # path to save the inpainted images
    save_path = "./images"

    # get scores
    # scores_telea = get_scores(dataset_path, save_path, (128, 128), loss_fns=[nn.MSELoss(), nn.L1Loss(), psnr], method="telea")
    # scores_ns = get_scores(dataset_path, save_path, (128, 128), loss_fns=[nn.MSELoss(), nn.L1Loss(), psnr], method="ns")

    # # get the average score
    # print("Average MSE score for Telea's algorithm: ", np.mean(scores_telea[0]))
    # print("Average MSE score for Navier-Stokes algorithm: ", np.mean(scores_ns[0]))
    # print("Average L1 score for Telea's algorithm: ", np.mean(scores_telea[1]))
    # print("Average L1 score for Navier-Stokes algorithm: ", np.mean(scores_ns[1]))
    # print("Average PSNR score for Telea's algorithm: ", np.mean(scores_telea[2]))
    # print("Average PSNR score for Navier-Stokes algorithm: ", np.mean(scores_ns[2]))

    scores_telea = get_scores(dataset_path, save_path, (128, 128), loss_fns=[fid], method="telea", save=True)
    scores_ns = get_scores(dataset_path, save_path, (128, 128), loss_fns=[fid], method="ns", save=True)

    # get the average score
    print("Average FID score for Telea's algorithm: ", np.mean(scores_telea[0]))
    print("Average FID score for Navier-Stokes algorithm: ", np.mean(scores_ns[0]))

