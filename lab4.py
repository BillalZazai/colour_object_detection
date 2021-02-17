import cv2
import numpy as np




print ('Lab 04\nBillal Zazai\n8572975')

trackbarName = 'trackbar'

trackbarWindow = 'result'
cv2.namedWindow(trackbarWindow)

def on_trackbar(val):
    print (val)
    HSV_img_clone = np.copy(HSV_img)
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    for i in range (image_height):
        for j in range (image_weight) :
            if ( (HSV_img_clone[i,j][0]<val) or (HSV_img_clone[i,j][0]>val) ): 
                HSV_img_clone[i,j] = [0,0,0]
    
    hsv_for = cv2.cvtColor(HSV_img_clone,cv2.COLOR_HSV2BGR)
    cv2.imshow(trackbarWindow, hsv_for)

# read an image using imread() function of cv2
# we have to  pass only the path of the image
picture1_name = 'Picture3.png'
img = cv2.imread('images/'+picture1_name)
# img = img.astype(np.float32)/255  

cv2.imshow('original image',img)

# converting the image into HSV format image
HSV_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

image_height, image_weight , image_channels = HSV_img.shape 

print (image_height, image_weight, image_channels)

# displaying the Hsv format image
cv2.imshow('HSV format image',HSV_img)


# hue_val = input ('Give me hue value: ')

cv2.createTrackbar( trackbarName, trackbarWindow , 0, 180 , on_trackbar)

on_trackbar (0)


# displaying the Hsv format image
# new_imae = HSV_img[0, 0]
# cv2.imshow('image',cv2.cvtColor( new_imae, cv2.COLOR_HSV2BGR))

# cv2.waitKey(0)
# Bitwise-AND mask and original image
# Threshold the HSV image to get only blue colors
#mask = cv2.inRange(HSV_img, lower_blue, upper_blue)


# res = cv2.bitwise_and(HSV_img,HSV_img, mask= mask)

# res = cv2.cvtColor(mask,cv2.COLOR_HSV2BGR)


# cv2.imshow('mask',mask)
# cv2.imshow('res', res)
cv2.waitKey()