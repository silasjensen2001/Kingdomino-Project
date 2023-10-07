import cv2
import numpy as np
from matplotlib import pyplot as plt
from ImageAnalyzer import ImageAnalyzer

path = "King Domino dataset\\Cropped and perspective corrected boards\\1.jpg"
img = cv2.imread(path)

#initialise analyzer object
analyzer = ImageAnalyzer()

tiles = analyzer.extract_tiles(img)

#cv.imshow("Image", tiles[11][1])

threshold = 0.39

##Read Main and Needle Image
imageMainRGB = tiles[18][1]
imageNeedleRGB = cv2.imread('crown3.png')
imageNeedleRGB.resize([30,24,3])

print(imageNeedleRGB.shape)

##Split Both into each R, G, B Channel
imageMainR, imageMainG, imageMainB = cv2.split(imageMainRGB)
imageNeedleR, imageNeedleG, imageNeedleB = cv2.split(imageNeedleRGB)

##Matching each channel
resultB = cv2.matchTemplate(imageMainR, imageNeedleR, cv2.TM_CCOEFF_NORMED)
resultG = cv2.matchTemplate(imageMainG, imageNeedleG, cv2.TM_CCOEFF_NORMED)
resultR = cv2.matchTemplate(imageMainB, imageNeedleB, cv2.TM_CCOEFF_NORMED)

##Add together to get the total score
result = resultB + resultG + resultR
print(result)
loc = np.where(result >= 3 * threshold)
print("loc: ", loc)

"""
assert tiles[23][1] is not None, "file could not be read, check with os.path.exists()"
img_gray = cv.cvtColor(tiles[23][1], cv.COLOR_BGR2GRAY)
template = cv.imread('Cropped Image.jpg', cv.IMREAD_GRAYSCALE)
template.resize([30,24])
assert template is not None, "file could not be read, check with os.path.exists()"

res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
threshold = 0.37
loc = np.where(res >= threshold)
"""
w = imageNeedleRGB.shape[0]
h = imageNeedleRGB.shape[1]

d_height = h/2
print(d_height)
for pt in zip(*loc[::-1]):
    img2 = cv2.rectangle(tiles[18][1], (pt[0], int(pt[1] + d_height)), (pt[0] + w, int(pt[1] + h + d_height)), (0,0,255), 2)
#print(pt)
#print(loc[::-1])
cv2.imwrite('res.png', img2)
