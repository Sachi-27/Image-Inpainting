import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

parser = argparse.ArgumentParser(description='New test script')

parser.add_argument('--image', type=str)
# parser.add_argument('--mask', type=str, default='center_mask_256.png', required=False)
# parser.add_argument('--checkpoint_path', type=str, help='Description of arg3', required=False)
# parser.add_argument('--output', type=str, help='Description of arg4')
parser.add_argument('--manual', type=bool)

args = parser.parse_args()

# mask_path = args.mask
image_path = args.image

mask = 0

if(args.manual):
    img = plt.imread(image_path)
    plt.imshow(img)
    plt.title("Click on 4 points")

    points = plt.ginput(4, timeout=0)

    # hull = ConvexHull(points)

    plt.close()

    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    x_coords, y_coords = zip(*points)

    plt.imshow(mask,cmap='gray')
    plt.axis('off')
    plt.fill(x_coords, y_coords, color='white')

    plt.savefig('new_mask.png', bbox_inches='tight', pad_inches=0)

    mask = cv2.imread('new_mask.png')

else:
    mask = cv2.imread('center_mask_256.png')


input = cv2.imread(image_path)
input_changed = cv2.resize(input,(256,256))
mask = cv2.resize(mask,(256,256))

mask = mask/255

mask_c = 1-mask
input_changed = input_changed*mask_c

input_changed = input_changed + mask*255

cv2.imwrite('masked_img.png',input_changed)






