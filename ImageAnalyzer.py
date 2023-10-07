import cv2
import numpy as np


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
        """

        tile_hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)

        thresholds = [[[24, 62], [82, 255], [72, 255]],
                      [[32, 43], [193, 255], [55, 255]],
                      [[94, 113], [146, 255], [0, 200]],
                      [[0, 63], [0, 160], [0, 94]],
                      [[32, 124], [0, 250], [0, 123]],
                      [[22, 45], [0, 178], [82, 255]]
                    ]

        results_list = []

        for thresh in thresholds:
            mask = cv2.inRange(tile_hsv, 
                              (thresh[0][0], thresh[1][0], thresh[2][0]),
                                (thresh[0][1], thresh[1][1], thresh[2][1]))
            
            mask1 = mask / 255
            result = mask1.sum() / mask1.size

            results_list.append(result)

        print(results_list)
        return np.argmax(results_list)
 

    def count_crowns(self, tile) -> int:
        pass

    def calculate_score(self, classification_list) -> int:
        pass