import cv2
import numpy as np
from matplotlib import pyplot as plt
from ImageAnalyzer import ImageAnalyzer

analyzer = ImageAnalyzer()

path = "King Domino dataset\\Cropped and perspective corrected boards\\1.jpg"
path2 = "ProcessedImages\Boards\ImageSharp.jpg"
img = cv2.imread(path2)
crop = img[212:234, 165:194]
tiles = analyzer.extract_tiles(img)

cv2.imshow("Image", tiles[18][1])


#Read Main and Needle Image
img = tiles[18][1]



kernel1 = np.array([[1,1,1,1,1],
                    [1,1,1,1,1],
                    [1,1,1,1,1],
                    [1,1,1,1,1],
                    [1,1,1,1,1]], np.uint8)

kernel2 = np.array([[1,1,1],
                    [1,1,1],
                    [1,1,1]], np.uint8)

def colour_threshold_BGR(image, name: str, lower_val: list, upper_val: list):
    # set lower and upper colour limits
    temp_lower_val = np.array(lower_val)
    temp_upper_val = np.array(upper_val)

    # Threshold the HSV image to get only green colours
    mask = cv2.inRange(image, temp_lower_val, temp_upper_val)

    # apply mask to original image - this shows the green with black blackground
    only_green = cv2.bitwise_and(image, image, mask = mask)

    # create a black image with the dimensions of the input image
    background = np.zeros(image.shape, image.dtype)
    # invert to create a white image
    background = cv2.bitwise_not(background)
    # invert the mask that blocks everything except green -
    # so now it only blocks the green area's
    mask_inv = cv2.bitwise_not(mask)
    # apply the inverted mask to the white image,
    # so it now has black where the original image had green
    masked_bg = cv2.bitwise_and(background, background, mask = mask_inv)
    # add the 2 images together. It adds all the pixel values, 
    # so the result is white background and the the green from the first image
    final = cv2.add(only_green, masked_bg)
    
    #show image
    #cv2.imshow(name, final)
    return final

def colour_threshold_HSV(image, name: str, lower_val: list, upper_val: list):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # set lower and upper colour limits
    temp_lower_val = np.array(lower_val)
    temp_upper_val = np.array(upper_val)

    # Threshold the HSV image to get only green colours
    mask = cv2.inRange(hsv, temp_lower_val, temp_upper_val)

    # apply mask to original image - this shows the green with black blackground
    only_green = cv2.bitwise_and(image, image, mask = mask)

    # create a black image with the dimensions of the input image
    background = np.zeros(image.shape, image.dtype)
    # invert to create a white image
    background = cv2.bitwise_not(background)
    # invert the mask that blocks everything except green -
    # so now it only blocks the green area's
    mask_inv = cv2.bitwise_not(mask)
    # apply the inverted mask to the white image,
    # so it now has black where the original image had green
    masked_bg = cv2.bitwise_and(background, background, mask = mask_inv)
    # add the 2 images together. It adds all the pixel values, 
    # so the result is white background and the the green from the first image
    final = cv2.add(only_green, masked_bg)
    
    #show image
    #cv2.imshow(name, final)
    return final


img_1 = colour_threshold_BGR(img, "image 1", [0, 125, 140], [121, 230, 235])
img_2 = colour_threshold_HSV(img, "image 2", [0, 50, 0], [177, 255, 255])


grayscaled_img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
grayscaled_img_2 = 60+grayscaled_img_2

(thresh, binary_img_2) = cv2.threshold(grayscaled_img_2, 60, 255, cv2.THRESH_BINARY)
binary_img_2 = 255-binary_img_2

dilation_1 = cv2.dilate(binary_img_2, kernel1, iterations=2)
erosion_1 = cv2.erode(dilation_1, kernel2, iterations=4)

img_3 = img_1+img_2

grayscaled_img_3 = cv2.cvtColor(img_3, cv2.COLOR_BGR2GRAY)
grayscaled_img_3 = grayscaled_img_3

(thresh, binary_img_3) = cv2.threshold(grayscaled_img_3, 250, 255, cv2.THRESH_BINARY)
binary_img_3 = 255-binary_img_3


#TEMPLATE MATCHING
cv2.imshow("crop", crop)
"""
assert img is not None, "file could not be read, check with os.path.exists()"
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#template = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
assert template is not None, "file could not be read, check with os.path.exists()"
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.3
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

cv2.imwrite('res.png',img)
"""







cv2.imshow("Thresholded BGR image", img_1)
cv2.imshow("Thresholded HSV image", img_2)
cv2.imshow("Thresholded HSV image 2", img_3)
cv2.imshow("Thresholded HSV image binary", binary_img_2)
cv2.imshow("Summed image", binary_img_3)
cv2.waitKey(0)
cv2.destroyAllWindows()



"""
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

###
assert tiles[23][1] is not None, "file could not be read, check with os.path.exists()"
img_gray = cv2.cvtColor(tiles[23][1], cv2.COLOR_BGR2GRAY)
template = cv2.imread('Cropped Image.jpg', cv2.IMREAD_GRAYSCALE)
template.resize([30,24])
assert template is not None, "file could not be read, check with os.path.exists()"

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.39
loc = np.where(res >= threshold)
###

w = imageNeedleRGB.shape[0]
h = imageNeedleRGB.shape[1]

#d_height = h/2
print(d_height)
for pt in zip(*loc[::-1]):
    img2 = cv2.rectangle(tiles[18][1], (pt[0], int(pt[1] + d_height)), (pt[0] + w, int(pt[1] + h + d_height)), (0,0,255), 2)
#print(pt)
#print(loc[::-1])
cv2.imwrite('res.png', img2)
"""