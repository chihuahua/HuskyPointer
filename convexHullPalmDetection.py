import numpy as np
import cv2

# Constants for finding range of skin color in YCrCb
min_YCrCb = np.array([0,133,77],np.uint8)
max_YCrCb = np.array([255,173,127],np.uint8)

cap = cv2.VideoCapture(0)                #creating camera object
while(cap.isOpened()):
  ret,img = cap.read()                      #reading the frames

  # Convert image to YCrCb
  imageYCrCb = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)

  # Find region with skin tone in YCrCb image
  skinRegion = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)

  # Do contour detection on skin region
  contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # find the best contours.
  if contours is None or not len(contours):
    continue

  max_area = 0
  for i in range(len(contours)):
    cnt=contours[i]
    area = cv2.contourArea(cnt)
    if(area>max_area):
      max_area=area
      ci=i
  cnt=contours[ci]

  # draw the convex hull.
  drawing = np.zeros(img.shape,np.uint8)

  hull = cv2.convexHull(cnt)

  cv2.drawContours(img,[hull],0,(0,255,0),2)

  hull = cv2.convexHull(cnt, returnPoints=False)

  if hull is None or cnt is None:
    continue

  defects = cv2.convexityDefects(cnt, hull)

  contour = cv2.approxPolyDP(cnt, 10, True)
  cv2.drawContours(img,[contour],0,(255,0,0),2)

  mind=0
  maxd=0
  i=0
  if defects is None:
    continue

  for i in range(min(defects.shape[0], 7000)):
    s,e,f,d = defects[i,0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv2.line(img,start,end,[0,255,0],2)
    cv2.circle(img,far,5,[0,0,255],-1)
    print(i)

  cv2.imshow('input', img) # displaying the frames
  k = cv2.waitKey(10)
  if k == 27:
    break

cap.release()
