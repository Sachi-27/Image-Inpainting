import cv2
import numpy as np
import random

# Define the dimensions of the mask and initialize it as a black image
mask_height = 300
mask_width = 400
mask = np.zeros((mask_height, mask_width), dtype=np.uint8)

# Define the number of white spots and the range of spot sizes
num_spots = 3  # Adjust as needed
min_spot_size = 10
max_spot_size = 50

# Create random white spots on the mask
for _ in range(num_spots):
    spot_size = random.randint(min_spot_size, max_spot_size)
    spot_x = random.randint(0, mask_width - spot_size)
    spot_y = random.randint(0, mask_height - spot_size)
    mask[spot_y:spot_y + spot_size, spot_x:spot_x + spot_size] = 255

# Save or display the mask image
cv2.imwrite('dataset/masks/random_spot_mask.png', mask)  # Save the mask image
cv2.imshow('Random Spot Mask', mask)  # Display the mask image (optional)

# Wait for a key press and then close the display window
cv2.waitKey(0)
cv2.destroyAllWindows()
