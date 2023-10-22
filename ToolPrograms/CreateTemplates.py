import numpy as np
import cv2
import sys
sys.path.append('../Kingdomino-Project')
from ImageProcessor import ImageProcessor

Processor = ImageProcessor() 

frame = cv2.imread('King Domino dataset\\Cropped and perspective corrected boards\\25.jpg') #hvis du vil se konceptet med et billede

#Preprocess image
img_equ = Processor.equalize_hist(frame)
img_sharp = Processor.sharpen_img(img_equ)

print("her")
print(img_sharp)

cv2.imwrite("ProcessedImages\Templates\TileTemplates\\Neutral4.jpg", img_sharp[120:160, 48:84])