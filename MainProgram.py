import cv2
import numpy as np
from ImageAnalyzer import ImageAnalyzer

path = "King Domino dataset\\Cropped and perspective corrected boards\\1.jpg"
img = cv2.imread(path)

#initialise analyzer object
analyzer = ImageAnalyzer()

tiles = analyzer.extract_tiles(img)

cv2.imshow("Image", tiles[11][1])







cv2.waitKey(0)
cv2.destroyAllWindows()



