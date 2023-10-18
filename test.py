import cv2
import numpy as np
from matplotlib import pyplot as plt
from ImageAnalyzer import ImageAnalyzer

analyzer = ImageAnalyzer()

path = "King Domino dataset\\Cropped and perspective corrected boards\\1.jpg"
path2 = "ProcessedImages\Boards\ImageSharp1.jpg"
img = cv2.imread(path2)
tiles = analyzer.extract_tiles(img)

#cv2.imshow("Image", tiles[18][1])

#read main image
img = tiles[1][1]

#crop_img = binary_img_3[14:35, 38:65] # cropped image from ImageSharp.jpg used to make the templates below
template_image_0 = cv2.imread("ProcessedImages//Templates//Template_0.jpg", cv2.IMREAD_GRAYSCALE)
template_image_90 = cv2.imread("ProcessedImages//Templates//Template_90.jpg", cv2.IMREAD_GRAYSCALE)
template_image_180 = cv2.imread("ProcessedImages//Templates//Template_180.jpg", cv2.IMREAD_GRAYSCALE)
template_image_270 = cv2.imread("ProcessedImages//Templates//Template_270.jpg", cv2.IMREAD_GRAYSCALE)

template_images = [template_image_0, template_image_90, template_image_180, template_image_270]

def colour_threshold(image, colour: str, lower_val: list, upper_val: list):
    if colour == "HSV":
        img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif colour == "BGR":
        img = image
    
    # Set the lower and upper colour limits
    temp_lower_val = np.array(lower_val)
    temp_upper_val = np.array(upper_val)

    # Threshold the HSV image to get the desired colours
    mask = cv2.inRange(img, temp_lower_val, temp_upper_val)

    # Apply the mask to the original image
    only_green = cv2.bitwise_and(image, image, mask = mask)

    # Create a black image that has the same dimensions as the input image
    background = np.zeros(image.shape, image.dtype)
    # Invert to create a white image
    background = cv2.bitwise_not(background)
    # Invert the mask that blocks everything except the desired colour, which results in only blocking that desired colour
    mask_inv = cv2.bitwise_not(mask)
    # Apply the inverted mask to the white image
    masked_bg = cv2.bitwise_and(background, background, mask = mask_inv)
    # Add the 2 images together
    final = cv2.add(only_green, masked_bg)

    return final

# [Pastures  ; Wheat Fields ; Lakes     ; Mines     ; Forests   ; Swamps   ]
# [0.32:0.77 ; 0.27:0.276  ; 0.34:0.82 ; 0.27:0.68 ; 0.36:0.76 ; 0.37:0.78 ]
def template_matching_with_rotated_templates(original_image, rotated_templates, threshold=0.27, overlap_threshold=0.9):
    img_1 = colour_threshold(original_image, "BGR", [0, 125, 140], [121, 230, 235])
    img_2 = colour_threshold(original_image, "HSV", [0, 50, 0], [177, 255, 255])
    img_3 = img_1+img_2
    
    #cv2.imshow("image1", img_1)
    #cv2.imshow("image2", img_2)
    #cv2.imshow("image3", img_3)

    grayscaled_img_3 = cv2.cvtColor(img_3, cv2.COLOR_BGR2GRAY)
    grayscaled_img_3 = grayscaled_img_3

    (thresh, binary_img_3) = cv2.threshold(grayscaled_img_3, 250, 255, cv2.THRESH_BINARY)

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
    cv2.imshow('res.png', result_image)
    
    return num_crowns


img_sample = np.array([[0, 2, 4, 4, 4],
                        [0, 4, 4, 4, 0],
                        [0, 5, 7, 4, 0],
                        [0, 5, 2, 0, 0],
                        [4, 2, 2, 0, 1]])

def grassfire(img):
    labeled_image = np.zeros_like(img)
    next_id = 1

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y, x] != 10 and labeled_image[y, x] == 0:
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

template_matching_with_rotated_templates(img, template_images)
#labeled_img = grassfire(img_sample)
#print(labeled_img)

# NEDENSTÅENDE FUNKTION ER IKKE FÆRDIG ENDNU
def count_score(input_array, info_list):
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























# MAIN
"""
img_1 = colour_threshold_BGR(img, "image 1", [0, 125, 140], [121, 230, 235])
img_2 = colour_threshold_HSV(img, "image 2", [0, 50, 0], [177, 255, 255])

img_3 = img_1+img_2
cv2.imshow("image1", img_1)
cv2.imshow("image2", img_2)
cv2.imshow("image3", img_3)
grayscaled_img_3 = cv2.cvtColor(img_3, cv2.COLOR_BGR2GRAY)
grayscaled_img_3 = grayscaled_img_3

(thresh, binary_img_3) = cv2.threshold(grayscaled_img_3, 250, 255, cv2.THRESH_BINARY)
dilated_binary_img_3 = cv2.dilate(binary_img_3, kernel=np.ones((3,3), np.uint8), iterations=1)
"""
#cv2.imshow("Binary image", dilated_binary_img_3)
#template_matching_with_rotated_templates(img, template_images)
cv2.waitKey(0)
cv2.destroyAllWindows()

