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

        
    def classify(self, cropped_img) -> list:
        """
        This function takes a cropped img as input, where only the board is visible and 
        returns a list ([[x,y], type (int)]) with classification of each tile.
        """
        tile_width, tile_height = 0


        

    def count_crowns(self, tile) -> int:
        pass

    def calculate_score(self, classification_list) -> int:
        pass