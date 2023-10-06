import cv2 
import matplotlib.pyplot as plt 
import random
import numpy as np

    


# reading image using the imread() function 
imageread = cv2.imread('images/dog.jpg') 
  
# input image is converted to gray scale image 
imagegray = cv2.cvtColor(imageread, cv2.COLOR_BGR2GRAY) 
  
# using the SIRF algorithm to detect key  
# points in the image 
features = cv2.SIFT_create() 
  
keypoints = features.detect(imagegray, None) 

# choose top 50 keypoints
keypoints = sorted(keypoints, key=lambda x: -x.response)[:50]
# randomly choose 5 keypoints
keypoints = random.sample(keypoints, 5)



# Load the input image
input_image = imageread

# Define the position (x, y) and size of the white spot

n=random.randint(2, 5)
mask = np.zeros_like(input_image)
for i in range(n):


    spot_x,spot_y = keypoints[i].pt
    spot_x = int(spot_x)
    spot_y = int(spot_y)
    spot_size = random.randint(10,25) # Size of the spot (adjust as needed)

    # Create a mask for the white spot (a white circle in this case)
    
    cv2.circle(mask, (spot_x, spot_y), spot_size, (255, 255, 255), thickness=-1)  # White circle

    # Add the white spot to the input image
    output_image = cv2.add(input_image, mask)

# Save or display the output image
# cv2.imwrite('output_image.jpg', output_image)  # Save the output image
cv2.imshow('Output Image', output_image)  # Display the output image (optional)

# Wait for a key press and then close the display window
cv2.waitKey(0)
cv2.destroyAllWindows()


# between every two keypoints draw a line
# for i in range(len(keypoints)):
#     for j in range(i+1, len(keypoints))[i:]:
#         x1, y1 = keypoints[i].pt
#         x2, y2 = keypoints[j].pt
#         # Draw line between two points of thickness 10 px
#         cv2.line(imageread, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), random.randint(1,10))

# plt.imshow(imageread)
  
# # drawKeypoints function is used to draw keypoints 
# # output_image = cv2.drawKeypoints(imageread, keypoints, 0, (255, 0, 0), 
#                                 #  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
  
# # displaying the image with keypoints as the 
# # output on the screen 
# # plt.imshow(output_image) 
  
# # plotting image 
# plt.show() 