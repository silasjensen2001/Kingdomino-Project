import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Define the path to the "Tiles" folder
tiles_folder1 = "ProcessedImages\\Tiles\\Neutral"
tiles_folder2 = "ProcessedImages\\Tiles\\Forest"
tiles_folder3 = "ProcessedImages\\Tiles\\Swamp"

folders = [tiles_folder1, tiles_folder2, tiles_folder3]

i = 1
j = 0
avg_cumsum = 0
avg_cumsums = []


# Plot all histograms in a single figure
fig, axs = plt.subplots(3, 3, figsize=(15, 10))
fig.suptitle('Histograms of H, S, and V channels for each picture type')

# Loop through each file in the "Tiles" folder
for tiles_folder in folders:
    Hs = []
    Ss = []
    Vs = []
    print("new")
    for filename in os.listdir(tiles_folder):
        # Check if the file is an image
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # Load the image
            image_path = os.path.join(tiles_folder, filename)
            image = cv2.imread(image_path)

            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            Hs.extend(hsv[:,:,0].flatten())
            Ss.extend(hsv[:,:,1].flatten())
            Vs.extend(hsv[:,:,2].flatten())
            
            print(len(Hs))

            # Apply a threshold to the image to isolate the small circles
            _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            thresh = 255 - thresh

            grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
            grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)

            img = cv2.bitwise_or(abs_grad_x, abs_grad_y)
            _, thresh_sobel = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
            
            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Count the number of small circles in the image
            small_circle_count = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 7 < area < 50:
                    cv2.drawContours(image, [contour], -1, (0,255,0), 3)
                    small_circle_count += 1
            
            # Append the score to the image filename
            score = small_circle_count

    
    axs[j, 0].hist(Hs, bins=20, color='r') #[i*10*10:(i+1)*10*10]
    axs[j, 0].set_title('{} Hue'.format(tiles_folder.split("\\")[-1]))
    axs[j, 1].hist(Ss, bins=20, color='g') #[i*10*10:(i+1)*10*10]
    axs[j, 1].set_title('{} Saturation'.format(tiles_folder.split("\\")[-1]))
    axs[j, 2].hist(Vs, bins=20, color='b') #[i*10*10:(i+1)*10*10]
    axs[j, 2].set_title('{} Value'.format(tiles_folder.split("\\")[-1]))
    
    #plt.show()

    j+=1

plt.show()



