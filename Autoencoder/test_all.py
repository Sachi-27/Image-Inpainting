import  numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import os, cv2, random

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

torch.manual_seed(40)
torch.cuda.manual_seed_all(40)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(f"{device}\n")

def read_image_tensor(image_folder,transform):
    torch.cuda.empty_cache()
    all_images = []
    for images in os.listdir(image_folder):
        img = torchvision.io.read_image(os.path.join(image_folder,images)).float()
        all_images.append(transform(img))
    print(f"Done with Animals")
    return torch.stack(all_images).to(device)


def get_mask():
    mask = np.zeros((128,128,3),dtype=np.float32)
    mask[32:96,32:96,:]=255
    return mask

def generate(img):

    min_num_spots, max_num_spots = 5, 8
    min_spot_size, max_spot_size = 10, 25
    black_img = np.zeros(img.shape)
    for _ in range(random.randint(min_num_spots, max_num_spots)):
        spot_size = random.randint(min_spot_size, max_spot_size)
        spot_x = random.randint(0, 256 - spot_size)
        spot_y = random.randint(0, 256 - spot_size)
        color = (255, 255, 255)  # White color in RGB format
        if random.choice([0,1]) == 1:
            # rectangle
            black_img[spot_y:spot_y + spot_size, spot_x:spot_x + spot_size] = color
            img[spot_y:spot_y + spot_size, spot_x:spot_x + spot_size] = color
        else:
            # circle
            cv2.circle(black_img, (spot_x, spot_y), spot_size, color, thickness=-1)
            cv2.circle(img, (spot_x, spot_y), spot_size, color, thickness=-1)

    return img,black_img

def make_input(labels):
    # apply generate input_image function on each image in labels and return the tensor of it
    images=[]

    mask = get_mask()
    mask[mask==255]=1
    for label in labels:
        img=np.array(label.cpu().permute(1,2,0).int())
        img = cv2.resize(img, (128, 128))
        # gen_img,mask= generate(img)
        gen_img = img*(1-mask)
        gen_img[gen_img==0]=255
        gen_img = np.array(torch.tensor(gen_img).cpu().permute(2,0,1).float())
        images.append(gen_img)
    mask[mask==1]=255
    mask = np.array(torch.tensor(mask).cpu().permute(2,0,1).float())

    return torch.tensor(np.array(images)).to(device),torch.tensor(mask).to(device)

img_size = (128,128)
base_transform = transforms.Compose(
    [transforms.Resize(img_size)
    ]
)

dataset = read_image_tensor("../animals_test", base_transform)
print(f"{dataset.shape}\n")

test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=32,
                                          shuffle=False)



from models import *
model = AutoEncoder(in_channels=3, out_channels=3)
model.to(device)

model = torch.load('lastmodel.pth', map_location=torch.device('cpu'))

k = 0
with torch.no_grad():
    for data in test_loader:
        data_inp = data.to(device)
        data,mask = make_input(data_inp)
        recon = model(data,mask)
        # Iterate over the data and recon and save the images
        for i in range(len(data)):
            plt.imsave(f"../animals_test_out/{k}_orig.png",data_inp[i].int().cpu().numpy().transpose((1, 2, 0)).astype(np.uint8))
            plt.imsave(f"../animals_test_out/{k}_mask.png",data[i].int().cpu().numpy().transpose((1, 2, 0)).astype(np.uint8))
            plt.imsave(f"../animals_test_out/{k}_recon.png",recon[i].int().cpu().numpy().transpose((1, 2, 0)).astype(np.uint8))
            k += 1
    print(f"Done with all images\n")



