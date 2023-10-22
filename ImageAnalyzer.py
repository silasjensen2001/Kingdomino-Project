import cv2
import numpy as np
from ImageProcessor import ImageProcessor
from matplotlib import pyplot as plt

class ImageAnalyzer:
    def __init__(self) -> None:
        #[[x,y], type (1,2,3..), crowns]
        self.classification_list = [[]]
        

    def extract_tiles(self, cropped_img) -> list:
        """
        This function takes a cropped img as input, where only the board is visible and
        returns a list ([[x,y], img]) with all the tiles.
        """
        tile_width = cropped_img.shape[1]//5
        tile_height = cropped_img.shape[0]//5
        extracted_tiles = []

        for i in range(5):
            for j in range(5):
                extracted_tiles.append([[i,j], cropped_img[i*tile_height : i*tile_height+tile_height,
                                                           j*tile_width : j*tile_width+tile_width,
                                                           :]])
        return extracted_tiles                                                        

        
    def classify_tile(self, tile) -> int:
        """
        This function takes a cropped image of a tile and classifies the type of the tile. 
        The type is defined from the following integers and thresholds as HSV
        0: Pastures (42-95, 0-255, 72-255)
        1: Wheat Fields (32-43, 193-255, 55-255)
        2: Lakes (94-113, 146-255, 0-200)
        3: Mines (0-63, 0-160, 0-94)
        4: Forests (32-124, 0-251, 0-123)
        5: Swamps (22-45, 0-178, 82-255)
        6: Table (10-21, 62-187, 0-241) 
        7: Other
        """

        tile_hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)

        thresholds = [[[36, 62], [98, 255], [125, 255]],
                      [[23, 28], [170, 255], [155, 255]],
                      [[94, 113], [146, 255], [0, 255]],
                      [[0, 255], [0, 150], [0, 50]], #[[0, 63], [0, 160], [0, 94]],
                      [[32, 124], [0, 150], [0, 123]], #[[32, 124], [0, 150], [0, 196]]
                      [[15, 31], [0, 178], [82, 255]],
                      [[10,21], [62,187], [0,241]],  #
                      [[15, 31], [0, 178], [82, 255]]
                    ]
        
        templates = [["Pasture1.jpg", "Pasture2.jpg"],
                     [],
                     [],
                     ["Mine1.jpg", "Mine2.jpg", "Mine3.jpg"],
                     ["Forest1.jpg"],
                     ["Swamp1.jpg"],
                     [],
                     ["Neutral1.jpg", "Neutral2.jpg", "Neutral3.jpg", "Neutral4.jpg"]]
        
        neutral_vs_swamp_thresh = [[0, 35],  [0, 255], [0,80]]
        forest_vs_mine_thresh =  [[10,33], [24,255], [150,255]]

        results_list = []

        for i, thresh in enumerate(thresholds):
            mask = cv2.inRange(tile_hsv, 
                              (thresh[0][0], thresh[1][0], thresh[2][0]),
                                (thresh[0][1], thresh[1][1], thresh[2][1]))
            
            mask1 = mask / 255
            result = mask1.sum() / mask1.size

            tile_gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            for j in range(len(templates[i])):
                #print(i)
                template = cv2.imread(f"ProcessedImages\Templates\TileTemplates\\{templates[i][j]}")
                template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

                for k in range(4):
                    template = cv2.rotate(template, cv2.ROTATE_90_CLOCKWISE)

                    res = cv2.matchTemplate(tile_gray, template, cv2.TM_CCOEFF_NORMED)
                    res = res*255

                    _, res_thresh = cv2.threshold(res, 150, 255, cv2.THRESH_BINARY)

                    res_thresh /= 255

                    if res_thresh.sum() > 0:
                        result += 1

                    #cv2.imshow("org", tile_gray)
                    #cv2.imshow("template", template)
                    #cv2.imshow("thresh", res_thresh)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()

                    #result += 



            results_list.append(result)


        #Check for neutral tiles. Too similar to swamps.
        if np.argmax(results_list) == 5:
            if max(results_list) < 0.3:
                return results_list, 7
            
            mask = cv2.inRange(tile_hsv, 
                              (neutral_vs_swamp_thresh[0][0], neutral_vs_swamp_thresh[1][0], neutral_vs_swamp_thresh[2][0]),
                                (neutral_vs_swamp_thresh[0][1], neutral_vs_swamp_thresh[1][1], neutral_vs_swamp_thresh[2][1]))
            
            mask = mask[10:90, 10:90]
            
            mask_eroded = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=2)
            mask_dilated = cv2.dilate(mask_eroded, np.ones((5,5), np.uint8), iterations=3)

            mask1 = mask_dilated / 255
            #print(mask1.sum())
            #cv2.waitKey()

            
            if mask1.sum() < 200:
                mask = cv2.inRange(tile_hsv, 
                            (forest_vs_mine_thresh[0][0], forest_vs_mine_thresh[1][0], forest_vs_mine_thresh[2][0]),
                                (forest_vs_mine_thresh[0][1], forest_vs_mine_thresh[1][1], forest_vs_mine_thresh[2][1]))
            
                mask = mask[10:90, 10:90]
                
                mask_eroded = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=2)
                mask_dilated = cv2.dilate(mask_eroded, np.ones((5,5), np.uint8), iterations=2)

                mask1 = mask_dilated / 255

                if mask1.sum() < 700:
                    return results_list, 7
                else:
                    return results_list, 5
                
        if np.argmax(results_list) == 3:
            if max(results_list) < 1:
                return results_list, 7
            else:
                return results_list, 3

        """    
        elif np.argmax(results_list) in [3,4]:
            mask = cv2.inRange(tile_hsv, 
                            (forest_vs_mine_thresh[0][0], forest_vs_mine_thresh[1][0], forest_vs_mine_thresh[2][0]),
                                (forest_vs_mine_thresh[0][1], forest_vs_mine_thresh[1][1], forest_vs_mine_thresh[2][1]))
            
            mask = mask[10:90, 10:90]
            
            mask_eroded = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=2)
            mask_dilated = cv2.dilate(mask_eroded, np.ones((5,5), np.uint8), iterations=2)

            mask1 = mask_dilated / 255
        
            if mask1.sum() < 750:
                return results_list, 4
            else:
                return results_list, 3
        """


        

        return results_list, np.argmax(results_list)
 

    def template_matching_with_rotated_templates(self, original_image, rotated_templates, threshold=0.37, overlap_threshold=0.9):
        # To find the crowns
        img_1 = ImageProcessor.colour_threshold(original_image, "HSV", [0, 50, 0], [177, 255, 255])
        #img_2 = ImageProcessor.colour_threshold(self, original_image, "BGR", [0, 125, 140], [121, 230, 235])
        #img_3 = img_1+img_2
        
        #cv2.imshow("image1", img_1)
        #cv2.imshow("image2", img_2)
        #cv2.imshow("image3", img_3)
        grayscaled_img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        #grayscaled_img_3 = cv2.cvtColor(img_3, cv2.COLOR_BGR2GRAY)

        _, binary_img_1 = cv2.threshold(grayscaled_img_1, 250, 255, cv2.THRESH_BINARY)
        #_, binary_img_3 = cv2.threshold(grayscaled_img_3, 250, 255, cv2.THRESH_BINARY)

        # Copy the original image as to be able to draw on it later
        result_image = original_image.copy()

        # Commence template matching with rotating templates
        matches = []
        num_crowns = 0

        for template in rotated_templates:
            res = cv2.matchTemplate(binary_img_1, template, cv2.TM_CCOEFF_NORMED)
            
            # Find matchede områder over tærskelværdien
            loc = np.where(res >= threshold)
            
            for pt in zip(*loc[::-1]):
                w, h = template.shape[::-1]
                # Tjek om dette resultat overlapper med tidligere resultater
                overlap = False
                for existing_match in matches:
                    dx = pt[0] - existing_match[0]
                    dy = pt[1] - existing_match[1]
                    distance = np.sqrt(dx * dx + dy * dy)
                    if distance < max(w, h) * overlap_threshold:
                        overlap = True
                        break

                # Only add the result if it doesn't overlap with previous results
                if not overlap:
                    matches.append(pt)
                    num_crowns += 1
                    cv2.rectangle(result_image, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
        
        #print(num_crowns)
        #cv2.imshow('res.png', result_image)
        
        return num_crowns

    
    def grassfire(self, img):
        labeled_image = np.zeros_like(img)
        next_id = 1

        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                if labeled_image[y, x] == 0: #img[y, x] != 10 and 
                    burn_queue = [(x, y)]
                    current_id = next_id

                    while burn_queue:
                        current_x, current_y = burn_queue.pop()
                        if img[current_y, current_x] == img[y, x]:
                            labeled_image[current_y, current_x] = current_id
                            # Check neighbors and add to the queue if they match the current value
                            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                                new_x, new_y = current_x + dx, current_y + dy
                                if (0 <= new_x < img.shape[1] and 0 <= new_y < img.shape[0] and labeled_image[new_y, new_x] == 0):
                                    burn_queue.append((new_x, new_y))

                    next_id += 1

        return labeled_image
    

    def count_score(self, input_array, info_list):
        final_score = 0
        number_of_interest = 0

        unique_numbers = np.max(input_array)

        while number_of_interest <= unique_numbers:
            value_list = []
            count = 0
            number_of_interest += 1
            
            for i in range(5):
                for j in range(5):
                    if input_array[i, j] == number_of_interest:
                        value_list.append(info_list[i,j])

            val = sum(value_list)*len(value_list)
            final_score += val

        return final_score