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






