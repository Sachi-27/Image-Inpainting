# Inpainting using classical methods
import numpy as np
import cv2

# Inpainting using Navier-Stokes algorithm
# https://docs.opencv.org/3.4/df/d3d/tutorial_py_inpainting.html
# https://docs.opencv.org/3.4/d7/d8b/group__photo__inpaint.html#gaedd30dfa0214fec4c88138b51d678085

# Inpainting using Telea's algorithm or the fast marching method
if __name__ == "__main__":
    # GENERATING INPUT and MASK
    orig_img = cv2.imread("images/dog.jpg")
    
    # Resize image to 256x256
    orig_img = cv2.resize(orig_img, (256, 256))
    # save the image
    cv2.imwrite("images/dog.jpg", orig_img)

    # We shall now add a small rectangles in the image with a mask of 255
    # This will be the region we shall inpaint
    img = orig_img.copy()
    img[50:70, 50:70, :] = 0
    img[100:130, 100:130, :] = 0
    img[150:170, 180:210, :] = 0
    img[80:100, 150:180, :] = 0
    cv2.imwrite("images/dog_input.jpg", img)

    # Also create the mask with same size 
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[50:70, 50:70] = 255
    mask[100:130, 100:130] = 255
    mask[150:170, 180:210] = 255
    mask[80:100, 150:180] = 255
    cv2.imwrite("images/dog_mask.jpg", mask)

    # Inpaint
    img = cv2.imread("images/dog_input.jpg")
    mask = cv2.imread("images/dog_mask.jpg", 0)

    # Inpaint using Navier-Stokes algorithm
    out_ns = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
    cv2.imwrite("images/dog_ns.jpg", out_ns)

    # Inpaint using Telea's algorithm
    out_telea = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    cv2.imwrite("images/dog_telea.jpg", out_telea)
