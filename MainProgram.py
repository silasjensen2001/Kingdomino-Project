import cv2
import numpy as np
from matplotlib import pyplot as plt
from ImageAnalyzer import ImageAnalyzer
from ImageProcessor import ImageProcessor

path = "King Domino dataset\\Cropped and perspective corrected boards\\1.jpg"
img = cv2.imread(path)

#initialise analyzer object
Analyzer = ImageAnalyzer()
Processor = ImageProcessor()


img_equ = Processor.equalize_hist(img)
img_sharp = Processor.sharpen_img(img_equ)

images = np.hstack([img, img_equ, img_sharp])
cv2.imshow("Equalized image", images)

tiles = Analyzer.extract_tiles(img_equ)
areas = Analyzer.classify_tile(tiles[1][1])

#vis alle billeder i matplot
for i in range(16):
    plt.subplot(4,4, i+1), plt.imshow(tiles[i][1][...,::-1], 'gray')
    plt.title("tile {}".format(i))
    plt.xticks([]), plt.yticks([])

plt.show()

#cv2.imshow("Image", tiles[2][1])
cv2.imwrite("ImageSharp.jpg", img_sharp)


cv2.waitKey(0)
cv2.destroyAllWindows()