import cv2
import numpy as np



class ImageProcessor:
    def __init__(self) -> None:
        pass

    def equalize_hist(self, img_BGR):
        # convert it to grayscale
        img_yuv = cv2.cvtColor(img_BGR,cv2.COLOR_BGR2YUV)

        # apply histogram equalization 
        #img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])

        #convert back
        hist_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        return hist_eq
    
    def sharpen_img(self, img):
        # applying kernels to the input image to get the sharpened image

        #Gaussian filter
        mean_filter = 1/16 * np.array([[1,2,1],
                                        [2,4,2],
                                        [1,2,1]])
        #mean_filter = 1/273 * np.array([[1,4,7,4,1],
         #                               [4,16,26,16,4],
          #                              [7,26,41,26,7],
           #                             [4,16,26,16,4],
            #                            [1,4,7,4,1]])
        
        img_avg = sharp_image = cv2.filter2D(img,-1,mean_filter)

        #Laplacian kernel
        sharpen_filter = np.array([[-1,-1,-1],
                                    [-1,9,-1],
                                    [-1,-1,-1]])
        
        sharp_image = cv2.filter2D(img_avg,-1,sharpen_filter)

        return sharp_image
    
    def colour_threshold_BGR(self, image, name: str, lower_val: list, upper_val: list):
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

    def colour_threshold_HSV(self, image, name: str, lower_val: list, upper_val: list):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
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






