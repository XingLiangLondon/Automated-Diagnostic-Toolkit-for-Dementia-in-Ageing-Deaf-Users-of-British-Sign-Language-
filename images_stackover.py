# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 10:28:02 2019

@author: liangx@westminster.ac.uk
"""

import cv2
import numpy as np
import os
file_root =("C:\\Users\\liangx\\Documents\\Github_Clone\\deep-learning-models\\data\\healthy\\" )
img_list=os.listdir(file_root)

for file_number in range (int(len(img_list)/2)):

    image1= cv2.imread(file_root+ str(file_number+1)+"_left2d_big.png" )
    image2= cv2.imread(file_root+ str(file_number+1)+"_right2d_big.png" )
    image3 =np.hstack((image1, image2))
    image4 =np.vstack((image1, image2))
    #cv2.imshow('combined image', image3)
    #cv2.waitKey()
    cv2.imwrite(file_root + str(file_number+1) + "_combine_2d_h.png", image3)
    cv2.imwrite(file_root + str(file_number+1) + "_combine_2d_v.png", image4)
