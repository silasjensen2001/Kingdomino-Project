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
        1: Pastures (42-95, 0-255, 72-255)
        2: Wheat Fields (32-43, 193-255, 55-255)
        3: Lakes (140-160, 0-255, 0-200)
        4: Mines (0-63, 0-160, 0-94)
        5: Forests (32-124, 0-251, 0-123)
        6: Swamps (22-45, 0-178, 82-255) 
        """

        tile_hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)

        l_b = np.array([l_h, l_s, l_v]) #nederst grænse for blå farve
        u_b = np.array([u_h, u_s, u_v]) #øverste grænse
        mask= cv2.inRange(hsv, l_b, u_b)
        res = cv2.bitwise_and(frame, frame, mask=mask)




        



        

    def count_crowns(self, tile) -> int:
        pass

    def calculate_score(self, classification_list) -> int:
        pass