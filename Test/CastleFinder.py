import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


# Creating the kernel(2d convolution matrix) 
kernel2 = np.array([[-1, 0, 1], 
                    [-1, 0, 1], 
                    [-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1]]) 


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

    for filename in os.listdir(tiles_folder):
        # Check if the file is an image
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # Load the image
            image_path = os.path.join(tiles_folder, filename)
            image = cv2.imread(image_path)

            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Apply a threshold to the image to isolate the small circles
            _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            thresh = 255 - thresh

            grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=7, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
            grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=7, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)

            img = cv2.bitwise_or(abs_grad_x, abs_grad_y)
            _, thresh_sobel = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

            # Applying the filter2D() function 
            img = cv2.filter2D(src=gray, ddepth=-1, kernel=kernel2) 

            img = np.abs(img)

            # Threshold the image
            _, thresh_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

            # Display the original image, grayscale image, and thresholded image
            cv2.imshow('Original', image) 
            cv2.imshow('Gray', gray) 
            cv2.imshow('Thresholded', thresh_sobel) 
            cv2.waitKey() 
            cv2.destroyAllWindows()
