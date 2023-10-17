import cv2
import numpy as np
from ImageProcessor import ImageProcessor

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

        thresholds = [[[36, 62], [82, 255], [141, 255]],
                      [[23, 28], [170, 255], [155, 255]],
                      [[94, 113], [146, 255], [0, 255]],
                      [[0, 63], [0, 160], [0, 94]],
                      [[32, 124], [0, 150], [0, 196]],
                      [[22, 31], [0, 178], [82, 255]],
                      [[10,21], [62,187], [0,241]],
                    ]
        
        neutral_vs_swamp_thresh = [[0, 29],  [179, 255], [0,130]]

        results_list = []

        for thresh in thresholds:
            mask = cv2.inRange(tile_hsv, 
                              (thresh[0][0], thresh[1][0], thresh[2][0]),
                                (thresh[0][1], thresh[1][1], thresh[2][1]))
            
            mask1 = mask / 255
            result = mask1.sum() / mask1.size

            results_list.append(result)


        #Check for neutral tiles. Too similar to swamps.
        if np.argmax(results_list) == 5:
            mask = cv2.inRange(tile_hsv, 
                              (neutral_vs_swamp_thresh[0][0], neutral_vs_swamp_thresh[1][0], neutral_vs_swamp_thresh[2][0]),
                                (neutral_vs_swamp_thresh[0][1], neutral_vs_swamp_thresh[1][1], neutral_vs_swamp_thresh[2][1]))
            mask1 = mask / 255
            if mask1.sum() < 800:
                return results_list, 7


        

        return results_list, np.argmax(results_list)
 

    def template_matching_with_rotated_templates(self, original_image, rotated_templates, threshold=0.37, overlap_threshold=0.9):
        img_1 = ImageProcessor.colour_threshold_BGR(self, original_image, "image 1", [0, 125, 140], [121, 230, 235])
        img_2 = ImageProcessor.colour_threshold_HSV(self, original_image, "image 2", [0, 50, 0], [177, 255, 255])
        img_3 = img_1+img_2
        
        #cv2.imshow("image1", img_1)
        #cv2.imshow("image2", img_2)
        #cv2.imshow("image3", img_3)

        grayscaled_img_3 = cv2.cvtColor(img_3, cv2.COLOR_BGR2GRAY)
        grayscaled_img_3 = grayscaled_img_3

        _, binary_img_3 = cv2.threshold(grayscaled_img_3, 250, 255, cv2.THRESH_BINARY)

        assert binary_img_3 is not None, "file could not be read, check with os.path.exists()"

        # Copy the original image as to be able to draw on it later
        result_image = original_image.copy()

        # Commence template matching with rotating templates
        matches = []
        num_crowns = 0

        for template in rotated_templates:
            res = cv2.matchTemplate(binary_img_3, template, cv2.TM_CCOEFF_NORMED)
            
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


    def calculate_score(self, classification_list) -> int:
        pass

    