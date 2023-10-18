import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
from ImageAnalyzer import ImageAnalyzer
from ImageProcessor import ImageProcessor


#initialise analyzer object
Analyzer = ImageAnalyzer()
Processor = ImageProcessor()

test_set_scores = []

#ALT MIT ROD
#crop_img = binary_img_3[14:35, 38:65] # cropped image from ImageSharp.jpg used to make the templates below
template_image_0 = cv2.imread("ProcessedImages//Templates//Template_0.jpg", cv2.IMREAD_GRAYSCALE)
template_image_90 = cv2.imread("ProcessedImages//Templates//Template_90.jpg", cv2.IMREAD_GRAYSCALE)
template_image_180 = cv2.imread("ProcessedImages//Templates//Template_180.jpg", cv2.IMREAD_GRAYSCALE)
template_image_270 = cv2.imread("ProcessedImages//Templates//Template_270.jpg", cv2.IMREAD_GRAYSCALE)

template_images = [template_image_0, template_image_90, template_image_180, template_image_270]


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

    classified_img = np.zeros((5,5))
    info_list = np.zeros((5,5))

    #vis alle billeder i matplot
    for i in range(25):
        result, type_idx = Analyzer.classify_tile(tiles[i][1])

        classified_img[i//5, i%5] = type_idx      
        
        if type_idx == 1:
            threshold = 0.32
        else:
            threshold = 0.45

        num_crowns = Analyzer.template_matching_with_rotated_templates(tiles[i][1], template_images, threshold)
        info_list[i//5, i%5] = num_crowns

        
        plt.subplot(5,5, i+1), plt.imshow(tiles[i][1][...,::-1], 'gray') #the funny tiles indexing converts the BGR color code to RGB
        plt.title(title_list[type_idx]+f' {num_crowns}')
        plt.xticks([]), plt.yticks([])

    img_blobs = Analyzer.grassfire(classified_img)
    final_score = Analyzer.count_score(img_blobs, info_list)

    print(final_score)
    
    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.05, 
                        hspace=0.5)
    
    plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()