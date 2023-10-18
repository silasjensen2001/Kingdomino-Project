import cv2
import numpy as np
import sys
sys.path.append('C:\\Users\\Thor9\\OneDrive - Aalborg Universitet\\Dokumenter\\AAU\\Kurser\\3-Semester\\Billedbehandling\\Kingdomino-Project')
from ImageProcessor import ImageProcessor

#path = f"King Domino dataset\\Cropped and perspective corrected boards\\14.jpg"
path = f"King Domino dataset\\Cropped and perspective corrected boards\\14.jpg"
img = cv2.imread(path)

Processor = ImageProcessor()

#Preprocess image
img_equ = Processor.equalize_hist(img)
img_sharp = Processor.sharpen_img(img_equ)

cv2.imshow("Image", img_sharp)
cv2.waitKey(0)
cv2.destroyAllWindows()