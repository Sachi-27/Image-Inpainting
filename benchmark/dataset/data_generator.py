import cv2
import numpy as np
import os
import random

def generate_mask():
    # Define the dimensions of the mask and initialize it as a black image
    mask_height = 256
    mask_width = 256
    mask = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)

    # Define the number of white spots and the range of spot sizes
    num_spots = 3  # Adjust as needed
    min_spot_size = 10
    max_spot_size = 50

    # Create random white spots on the mask
    for _ in range(num_spots):
        spot_size = random.randint(min_spot_size, max_spot_size)
        spot_x = random.randint(0, mask_width - spot_size)
        spot_y = random.randint(0, mask_height - spot_size)
        color = (255, 255, 255)  # White color in RGB format
        mask[spot_y:spot_y + spot_size, spot_x:spot_x + spot_size] = color
    # cv2.imshow('Random Spot Mask', mask)  # Display the mask image (optional)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return mask

def apply_mask_to_image(image_dir, mask_img, output_dir):
    for dir in os.listdir(image_dir):
        for image in os.listdir(os.path.join(image_dir, dir)):
            img = cv2.imread(os.path.join(os.path.join(image_dir,dir), image))
            # cv2.imshow('Random Spot Mask', img)  # Display the mask image (optional)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            img = cv2.resize(img, (256, 256))
            # print(img.shape)
            # mask = cv2.imread(mask_img)
            # print(mask.shape)
            mask = generate_mask()
            mask = cv2.bitwise_not(mask)
            masked_image = cv2.bitwise_and(img, mask)
            # change the color of the masked image to white
            masked_image[mask == 0] = 255
            cv2.imwrite(os.path.join(output_dir, image), masked_image)


if __name__ == "__main__":
    image_dir = "dataset/animals"
    mask = "dataset/masks/dog_mask.jpg"
    output_dir = "dataset/inputs"
    apply_mask_to_image(image_dir, mask, output_dir)