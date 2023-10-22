import cv2
import numpy as np
import random

num_test_imgs = 15
nums = []
num = random.randint(0, 74)

for i in range(15):
    while num in nums:
        num = random.randint(0, 74)
    
    nums.append(num)
    img = cv2.imread(f'King Domino dataset\Cropped and perspective corrected boards\\{num}.jpg')

    cv2.imwrite(f'King Domino dataset\TestSet\Img{i}.jpg', img)

