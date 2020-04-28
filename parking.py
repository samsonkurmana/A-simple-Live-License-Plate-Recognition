# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:33:37 2020

@author: dell
"""

import cv2
import numpy as np
import time,os
import pytesseract as tess
import string
import pandas as pd

currdir=os.getcwd()

exemption=pd.read_csv("vehicle.csv")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

#######################################################

import cv2

def find_contours(dimensions, img) :

    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    
    # Check largest 5 or  15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    
    ii = cv2.imread('contour.jpg')
    
    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :
        #detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        #checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((44,24))
            #extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            
            cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
            plt.imshow(ii, cmap='gray')

#             Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy) #List that stores the character's binary image (unsorted)
            
    #Return characters on ascending order with respect to the x-coordinate (most-left character first)
            
    plt.show()
    #arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])# stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res

import pymysql

import matplotlib.pyplot as plt

def segment_characters(image) :

    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (5,5))
    img_binary_lp = cv2.dilate(img_binary_lp, (5,5))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/6,
                       LP_WIDTH/2,
                       LP_HEIGHT/10,
                       2*LP_HEIGHT/3]
    #plt.imshow(img_binary_lp, cmap='gray')
    plt.show()
    cv2.imwrite('contour.jpg',img_binary_lp)

    # Get contours within cropped license plate
    #char_list = find_contours(dimensions, img_binary_lp)

#    return char_list
    return img_binary_lp







#############################################################


tess.pytesseract.tesseract_cmd =r''+currdir+'\\Tesseract-OCR\\Tesseract-OCR\\tesseract.exe'





##########################



file="L.mp4"



#############################





plate_cascade = cv2.CascadeClassifier('indian_license_plate.xml')
#######if file exists filename else '0' for live capture
cap=cv2.VideoCapture(file)

########## For live video capture############

#cap=cv2.VideoCapture(0)



#####setting up frames

cap.set(cv2.CAP_PROP_FPS, int(300))
kernel = np.ones((5,5),np.uint8)


###Connection with database
def sqld(query):
    quer="select name from cars where number like '%" +query+"%'"
    #print(quer)
    quer=quer
    cnx = pymysql.connect(user='root', password='12345',
                                  host='127.0.0.1',
                                  database='parking')
    cur = cnx.cursor()
    cur.execute(quer) 
        #data=cur.fetchall()
    data=cur.fetchone()
    return(data) 
    


#### Checking with the list of authorised people access numbers
def checklis(text):
    try:
       return exemption.loc[exemption['Vehicle'] == str(text), 'Name'].iloc[0]
    except IndexError:
       pass
            
    
         
        



# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
     frame1= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
     plate_rect = plate_cascade.detectMultiScale(frame, 1.2, 4)
     
      
     for (x,y,w,h) in plate_rect:
        cv2.rectangle(frame1, (x,y), (x+w, y+h), (0,0,255), 3)
        plate=frame[y:y+h, x:x+w]
        
        ###################
        #nplate=cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
        #opening = cv2.morphologyEx(nplate, cv2.MORPH_OPEN, kernel)
        #dilation = cv2.dilate(plate,kernel,iterations = 1)
                
        ###################
        char=segment_characters(plate)
#        text=tess.image_to_string(char,config="-c tessedit_char_whitelist=%s_-." % char_whitelist)
        text=tess.image_to_string(char,config="-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 6")
        font=cv2.FONT_HERSHEY_SIMPLEX
        
        text=str(text)
        h=checklis(text)
        #continue
       
        
        frame=cv2.putText(frame,h,(10,50),font,1,(0,255,255),2,cv2.LINE_AA)
    # Display the resulting frame
     cv2.imshow('Frame',frame)
     #cv2.imshow('plate',plate)
     out.write(frame)
     cv2.imwrite('plate.png',plate)


    # Press Q on keyboard to  exit
     if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()
out.release()
# Closes all the frames
cv2.destroyAllWindows()