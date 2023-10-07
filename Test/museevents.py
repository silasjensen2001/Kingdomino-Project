import numpy as np
import cv2

#events = [i for i in dir(cv2) if 'EVENT' in i] iterer igennem cv2 pakkens mouseevnts og danne en array
#print(events)

def click_event(event, x, y, flags, param):
    #printer koordinaterne ved venstreklik
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,', ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY = str(x) + ', ' + str(y) #str(x) laver det om til en string
        cv2.putText(img, strXY, (x, y), font, 0.5, (255, 255, 0), 2)
        cv2.imshow('image', img)
    #printer BGR ved højreklik    
    if event == cv2.EVENT_RBUTTONDOWN:
        blue = img[y, x, 0]
        green = img[y, x, 1]
        red = img[y, x, 2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        strBGR = str(blue) + ', ' + str(green) + ', ' + str(red) #str(x) laver det om til en string
        cv2.putText(img, strBGR, (x, y), font, 0.5, (0, 255, 0), 2)
        cv2.imshow('image', img)

#img = np.zeros((512,512,3), np.uint8)
img = cv2.imread('King Domino dataset\\Cropped and perspective corrected boards\\1.jpg')
cv2.imshow('image', img) #vinduenavnet skal være ens

cv2.setMouseCallback('image', click_event) #kalder vores funktion ved museaktivitet

cv2.waitKey(0)
cv2.destroyAllWindows()

