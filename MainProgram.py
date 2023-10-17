import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
from ImageAnalyzer import ImageAnalyzer
from ImageProcessor import ImageProcessor


#initialise analyzer object
Analyzer = ImageAnalyzer()
Processor = ImageProcessor()

special_tiles = [13,13,13,13,23,13,13,13,23,20,13,15,23,20,13,15,23,13]

for j in range(1, 50):
    path = f"King Domino dataset\\Cropped and perspective corrected boards\\{j}.jpg"
    img = cv2.imread(path)

    #Preprocess image
    img_equ = Processor.equalize_hist(img)
    img_sharp = Processor.sharpen_img(img_equ)

    #images = np.hstack([img, img_equ, img_sharp])
    #cv2.imshow("Equalized image", images)

    #Divide board image into multiple tile images and classify them
    tiles = Analyzer.extract_tiles(img_sharp)

    title_list = ["Pastures", "Wheat", "Lakes", "Mines", "Forests", "Swamps", "Table", "Other"]

    #Random integers that determines which tiles to save
    rand_ints = [random.randint(0, 24) for i in range(2)]

    print(rand_ints)
    #vis alle billeder i matplot
    for i in range(25):
        result, type_idx = Analyzer.classify_tile(tiles[i][1])

        plt.subplot(5,5, i+1), plt.imshow(tiles[i][1][...,::-1], 'gray') #the funny tiles indexing converts the BGR color code to RGB
        plt.title(title_list[type_idx])
        plt.xticks([]), plt.yticks([])

        #Uncomment to retrieve random tiles
        #if i in rand_ints:
        #   cv2.imwrite(f"ProcessedImages\\Tiles\\Mixed\\Tile{j}_{i}.jpg", tiles[i][1])

        #Uncomment to retrieve neutral tiles
        #if i+1 == special_tiles[j]:
        #   cv2.imwrite("ProcessedImages\\Tiles\\Neutral\\Tile{}.jpg".format(j), tiles[i][1])

    
    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.05, 
                        hspace=0.5)
    
    plt.show()

"""
    for i in range(9):
        result, type_idx = Analyzer.classify_tile(tiles[i+16][1])
        plt.subplot(4,4, i+1), plt.imshow(tiles[i+16][1][...,::-1], 'gray')
        plt.title(title_list[type_idx])
        plt.xticks([]), plt.yticks([])

        #cv2.imwrite("ProcessedImages\\Boards\\Tile{}.jpg".format(i), tiles[i][1])

    plt.show()
"""


#cv2.imshow("Image", tiles[2][1])
#cv2.imwrite("ImageSharp1.jpg", img_sharp)


cv2.waitKey(0)
cv2.destroyAllWindows()