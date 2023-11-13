import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def set_seed():
    torch.manual_seed(123)
    np.radom.seed(42)


def psnr(img1, img2):
    set_seed()
    mse = nn.MSELoss()(img1, img2)
    if mse == 0:
        print("MSE is 0")
        return 100
    # max pixel value
    PIXEL_MAX = np.max(img1.numpy())
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))




# scale an array of images to a new size
def scale_images(images, new_shape):
    set_seed()
    images_list = list()
    for image in images:
    # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return np.asarray(images_list)

# The FID (Fréchet Inception Distance) score is commonly calculated using the InceptionV3 model, 
# The choice of the InceptionV3 model is based on its effectiveness in capturing features relevant to 
# image quality and diversity. 
# The FID score compares the statistics of feature representations (specifically, the activations in one of the intermediate layers) 
# of real and generated images.
# Reference: https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/#:~:text=The%20Frechet%20Inception%20Distance%20score,for%20real%20and%20generated%20images.

def fid(output_imgs, gt_imgs):
    set_seed()

    # Ensure input images1 and images2 are of shape (n_images, height, width, 3)
    
    # Convert tensor to numpy
    images1 = output_imgs.numpy()
    images2 = gt_imgs.numpy()
    # prepare the inception v3 model
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
    # Convert integer to float
    images1 = images1.astype('float32')
    images2 = images2.astype('float32')
    # resize images
    images1 = scale_images(images1, (299,299,3))
    images2 = scale_images(images2, (299,299,3))
    # pre-process images
    images1 = preprocess_input(images1)
    images2 = preprocess_input(images2)

    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


## LPIPS - Learned Perceptual Image Patch Similarity
# Reference: https://torchmetrics.readthedocs.io/en/stable/image/learned_perceptual_image_patch_similarity.html
# A low LPIPS score means that image patches are perceptual similar.
# Both input image patches are expected to have shape (N, 3, H, W). 
# The minimum size of H, W depends on the chosen backbone (see net_type arg).
def lpips(output_imgs, gt_imgs):
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze', reduction='mean')
    return lpips(output_imgs, gt_imgs).item()
 
