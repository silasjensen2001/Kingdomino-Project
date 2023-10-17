import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('C:\\Users\\Thor9\\OneDrive - Aalborg Universitet\\Dokumenter\\AAU\\Kurser\\3-Semester\\Billedbehandling\\Kingdomino-Project')
from ImageProcessor import ImageProcessor
import math


Processor = ImageProcessor()


# Define the path to the "Tiles" folder
tiles_folder1 = "ProcessedImages\\Tiles\\Neutral"
tiles_folder2 = "ProcessedImages\\Tiles\\Forest"
tiles_folder3 = "ProcessedImages\\Tiles\\Swamp"

folders = [tiles_folder1, tiles_folder2, tiles_folder3]


# Loop through each file in the "Tiles" folder
for tiles_folder in folders:
    for filename in os.listdir(tiles_folder):
        # Check if the file is an image
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # Load the image
            image_path = os.path.join(tiles_folder, filename)
            image = cv2.imread(image_path)

            img_equ = Processor.equalize_hist(image)
            img_sharp = Processor.sharpen_img(img_equ)

            # Convert the image to grayscale
            gray = cv2.cvtColor(img_sharp, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img_sharp, cv2.COLOR_BGR2HSV)

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

            edges = cv2.Canny(thresh,100,200)

            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Taking a matrix of size 5 as the kernel 
            kernel = np.ones((3, 3), np.uint8) 
             
            img_erosion = cv2.erode(thresh, kernel, iterations=1) 
            img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)

            for contour in contours:
                area = cv2.contourArea(contour)
                if 5 < area < 50:
                    cv2.drawContours(img_sharp, [contour], -1, (0,255,0), 1)


            # Display the original image, grayscale image, and thresholded image
            cv2.imshow('Original', img_sharp) 
            cv2.imshow('Threshed', img_dilation) 
            cv2.imshow('Thresholded', edges) 
            #cv2.imshow('Hue channel', img_hue) 
            cv2.waitKey() 
            cv2.destroyAllWindows()
