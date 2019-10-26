# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:59:52 2019

@author: Imsourav
"""
import imutils
import cv2
import shutil
from Prediction import *
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
def edge_cut(outputname):
    img = cv2.imread(outputname) 
    rsz_img = img # resize since image is huge
    gray = cv2.cvtColor(rsz_img, cv2.COLOR_BGR2GRAY) # convert to grayscale

# threshold to get just the signature
    retval, thresh_gray = cv2.threshold(gray, thresh=100, maxval=255, type=cv2.THRESH_BINARY)

# find where the signature is and make a cropped region
    points = np.argwhere(thresh_gray==0) # find where the black pixels are
    points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices
    x, y, w, h = cv2.boundingRect(points) # create a rectangle around those points
    if y>10 and x>15:
        x, y, w, h = x-15, y-10, w+20, h+15 # make the box a little bigger
    crop = gray[y:y+h, x:x+w] # create a cropped region of the gray image

# get the thresholded crop
    retval, thresh_crop = cv2.threshold(crop, thresh=200, maxval=255, type=cv2.THRESH_BINARY)

# display
                #cv2.imshow("Cropped and thresholded image", thresh_crop) 
    cv2.imwrite("0"+outputname,thresh_crop)   
    
def crop(infile,temp,output):
    
    prev=0
    im6 = Image.open(infile)
    imgwidth12, imgheight13 = im6.size
    for i in range(10,imgheight13,5):
        im = Image.open(infile)
        imgwidth, imgheight = im.size
        im1 = im.crop((1, 1,imgwidth, i)) 
        im1.save('kalu.jpg')
        edge_cut('kalu.jpg')
        im2 = Image.open('kalu.jpg')
        imgwidth1, imgheight1 = im2.size
       
        try:
            im3 = Image.open("0"+"kalu.jpg")
            imgwidth2, imgheight2 = im3.size
           # print(imgheight1 , i)
            #print(imgheight2)
            if prev==imgheight2 and imgheight2>10:
                print(i)
                im4 = Image.open(infile)
                im4 = im4.crop((1,i,imgwidth, imgheight)) 
                im4.save('sou.jpg')
                st="Demo/store/"+temp                
                shutil.copy("0"+"kalu.jpg",st)
                str12=st
               # cv2.namedWindow("output", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
                #im = cv2.imread(str12)                        # Read image
                #imS = cv2.resize(im, (2060, 740))                    # Resize image
                #cv2.imshow("output", imS)
                how(st,output)
                return 1
            prev=imgheight2 
        except:
            a1=0
          
    return 0

    
  
#if __name__== "__main__" :
def call1(input1,output):
    open(output,'w').close()
    open('output/output3.txt', 'w').close()
    open('output/output2.txt', 'w').close()
    open('output/output1.txt', 'w').close()
    open('output/output4.txt', 'w').close()
    temp=".jpg"
    cnt=1
    shutil.copy(input1,"sou.jpg")
    while True:
        temp1=str(cnt)+temp
        a=crop('sou.jpg',temp1,output)
        cnt+=1
        if a==0:
            break
   
    
    
    