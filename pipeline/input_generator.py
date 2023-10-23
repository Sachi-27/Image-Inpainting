import os, cv2, random
import pyttsx3

# Generates input image which is a square bob at center of image with 1/4 size
def generate_input_image1(ing):
    img_size = img.shape[0]
    x1=y1=int(img_size/4)
    x2=y2=int(img_size*3/4)
    img[x1:x2, y1:y2] = (255, 255, 255)
    return img


# Generates input image which is randomly generated lines and spots
def generate_input_image2(img):
    # SIRF algorithm to detect key points in the image
    print(img.shape)
    features = cv2.SIFT_create()
    keypoints = features.detect(img, None)
    # Choose top 50 and bottom 50 keypoints
    keypoints1 = sorted(keypoints, key=lambda x: -x.response)[:min(50, len(keypoints))]
    keypoints2 = sorted(keypoints, key=lambda x: x.response)[:min(50, len(keypoints))]
    # Randomly choose 3 keypoints from each set
    keypoints1 = random.sample(keypoints1, 3)
    keypoints2 = random.sample(keypoints2, 3)
    
    # between every two keypoints draw a line
    for i in range(len(keypoints1)):
        for j in range(len(keypoints2)):
            x1, y1 = keypoints1[i].pt
            x2, y2 = keypoints2[j].pt
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), random.randint(1,8))
    

    #### 
    # Add random white spots to the image
    min_num_spots, max_num_spots = 2, 5
    min_spot_size, max_spot_size = 10, 25
    for _ in range(random.randint(min_num_spots, max_num_spots)):
        spot_size = random.randint(min_spot_size, max_spot_size)
        spot_x = random.randint(0, 256 - spot_size)
        spot_y = random.randint(0, 256 - spot_size)
        color = (255, 255, 255)  # White color in RGB format
        if random.choice([0,1]) == 1:
            # rectangle
            img[spot_y:spot_y + spot_size, spot_x:spot_x + spot_size] = color
        else:
            # circle
            cv2.circle(img, (spot_x, spot_y), spot_size, color, thickness=-1)

    return img

if __name__ == "__main__":
    image_dir = "../animals/animals"
    output_dir = "../animals/input_animals"
    # If output directory does not exist, create it
    engine = pyttsx3.init()

    # AUDIO MESSAGES
    engine.say("Rom Rom Bhaaiyo")
    engine.runAndWait()


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for subdir in os.listdir(image_dir):
        # If output subdirectory does not exist, create it
        if not os.path.exists(os.path.join(output_dir, subdir)):
            os.makedirs(os.path.join(output_dir, subdir))
        for image in os.listdir(os.path.join(image_dir, subdir)):
            img = cv2.imread(os.path.join(image_dir, subdir, image))
            shape = img.shape
            img = cv2.resize(img, (256, 256))
            output = generate_input_image2(img)
            # reshape to original size
            output = cv2.resize(output, (shape[1], shape[0]))
            cv2.imwrite(os.path.join(output_dir, subdir, image), output)
        print("Done with", subdir)
        # AUDIO MESSAGES
        engine.say("Done with "+subdir)
        engine.runAndWait()
    
    for i in range(3):
        print("Jai Shree Raam")
        # AUDIO MESSAGES
        engine.say("Jai Shree Raam")
        engine.runAndWait()
        
