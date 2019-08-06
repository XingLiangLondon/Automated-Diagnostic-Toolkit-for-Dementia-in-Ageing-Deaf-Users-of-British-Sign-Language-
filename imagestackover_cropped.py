# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 10:28:02 2019

@author: liangx@westminster.ac.uk
"""

import cv2
import numpy as np
import os
file_root =("C:\\Users\\liangx\\Documents\\Github_Clone\\deep-learning-models\\data\\dementia\\" )
img_list=os.listdir(file_root)

#get the height and width of a sample image 
image =cv2.imread (file_root + "1_right2d_big.png" ) 
height, width=image.shape[:2]


for file_number in range (int(len(img_list)/2)):

    image1= cv2.imread(file_root+ str(file_number+1)+"_left2d_big.png" )
    image2= cv2.imread(file_root+ str(file_number+1)+"_right2d_big.png" )
    # crop the image (get rid of white margin)
    image1=image1[80:(height-60), 140:(width-140)]
    image2=image2[80:(height-60), 140:(width-140)]
    image3 =np.hstack((image1, image2))
    image4 =np.vstack((image1, image2))
    #cv2.imshow('combined image', image3)
    #cv2.waitKey()
    cv2.imwrite(file_root + str(file_number+1) + "_combine_2d_h.png", image3)
    cv2.imwrite(file_root + str(file_number+1) + "_combine_2d_v.png", image4)
    
    
    
    
#image =cv2.imread (file_root + "5_right2d_big.png" ) 
#height, width=image.shape[:2]
#image=image[80:(height-60), 140:(width-140)]
#cv2.imshow("image", image)   

#cv2.waitKey() #13 is the Enter Key
#cv2.destroyAllWindows()  
