import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('C:\\Users\\Thor9\\OneDrive - Aalborg Universitet\\Dokumenter\\AAU\\Kurser\\3-Semester\\Billedbehandling\\Kingdomino-Project')
from ImageProcessor import ImageProcessor
import math




Processor = ImageProcessor()


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

            grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
            grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)

            img = cv2.bitwise_or(abs_grad_x, abs_grad_y)
            _, thresh_sobel = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

            kernel_blur = np.ones((5,5),np.float32)/25
            img_blur = cv2.filter2D(gray,-1,kernel_blur)

            # Applying the filter2D() function 
            #img = cv2.filter2D(src=hsv[:,:,1], ddepth=-1, kernel=kernel2) 
            #_, thresh_sobel = cv2.threshold(hsv[:,:,1], 150, 255, cv2.THRESH_BINARY)

            edges = cv2.Canny(img_blur,100,200)


            """
            # Standard Hough Line Transform
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 50, None, 0, 0)

             # Draw the lines
            if lines is not None:
                for i in range(0, len(lines)):
                    rho = lines[i][0][0]
                    theta = lines[i][0][1]
                    a = math.cos(theta)
                    b = math.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                    cv2.line(image, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
            """

            
            # Probabilistic Line Transform
            linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, None, 25, 5)

            if linesP is not None:
                for i in range(0, len(linesP)):
                    l = linesP[i][0]
                    cv2.line(image, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

            

            # Threshold the image
            _, thresh_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

            #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            #img_hue = clahe.apply(hsv[:,:,0])

            # Display the original image, grayscale image, and thresholded image
            cv2.imshow('Original', image) 
            cv2.imshow('Blurred', img_blur) 
            cv2.imshow('Thresholded', edges) 
            #cv2.imshow('Hue channel', img_hue) 
            cv2.waitKey() 
            cv2.destroyAllWindows()
