# Task1
# detect each of the yellow shapes on the video frames. Classify the shapes into
# classes: circle, rectangle, triangle. Make an output video with each of the shapes signed by
# the name of its class. Mark the classes with different colors. The shapes which are partly
# visible may be ignored.

import cv2
import numpy as np

def empty(a):
  pass

TrackBars = "TrackBars"
cv2.namedWindow(TrackBars)
cv2.resizeWindow(TrackBars, 840, 640)
cv2.createTrackbar("Hue Min", TrackBars, 80, 179, empty)
cv2.createTrackbar("Hue Max", TrackBars, 137, 179, empty)
cv2.createTrackbar("Sat Min", TrackBars, 0, 255, empty)
cv2.createTrackbar("Sat Max", TrackBars, 254, 255, empty)
cv2.createTrackbar("Val Min", TrackBars, 8, 255, empty)
cv2.createTrackbar("Val Max", TrackBars, 255, 255, empty)
cv2.createTrackbar("Canny Min", TrackBars, 7, 255, empty)
cv2.createTrackbar("Canny Max", TrackBars, 13, 255, empty)
cv2.createTrackbar("Min Area", TrackBars, 3500, 200000, empty)
cv2.createTrackbar("dialation kernal", TrackBars, 5, 255, empty)
cv2.createTrackbar("dialation iterations", TrackBars, 5, 255, empty)


def stackImages(scale, imgArray):
  rows = len(imgArray)
  cols = len(imgArray[0])
  rowsAvailable = isinstance(imgArray[0], list)
  width = imgArray[0][0].shape[1]
  height = imgArray[0][0].shape[0]
  if rowsAvailable:
    for x in range(0, rows):
      for y in range(0, cols):
        if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
          imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
        else:
          imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale,
                                      scale)
        if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
    imageBlank = np.zeros((height, width, 3), np.uint8)
    hor = [imageBlank] * rows
    hor_con = [imageBlank] * rows
    for x in range(0, rows):
      hor[x] = np.hstack(imgArray[x])
    ver = np.vstack(hor)
  else:
    for x in range(0, rows):
      if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
        imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
      else:
        imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
      if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
    hor = np.hstack(imgArray)
    ver = hor
  return ver

def getCountours(img, imgContours):
  contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  # print(len(contours))
  for cnt in contours:
    area = cv2.contourArea(cnt)
    area_min = cv2.getTrackbarPos("Min Area", TrackBars)
    if area > area_min:
      peri = cv2.arcLength(cnt, True)
      approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
      print(len(approx))
      objCorn = len(approx)
      x_, y_, w_, h_ = cv2.boundingRect(approx)

      objColar = (0, 0, 0)
      if objCorn == 3:
        objectType = "Triangle"
        objColar = (255, 0, 0)
      elif objCorn == 4:
          objectType = "Rectangle"
          objColar = (0, 255, 0)
      elif objCorn > 4:
        objectType = "Circles"
        objColar = (0, 0, 255)
      else:
        objectType = "None"

      cv2.rectangle(imgContours, (x_,y_), (x_ + w_, y_ + h_), objColar, 5)
      cv2.putText(imgContours, objectType, (x_, y_), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 1)



def сolor_detection(img):
  imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  h_min = cv2.getTrackbarPos("Hue Min", TrackBars)
  h_max = cv2.getTrackbarPos("Hue Max", TrackBars)
  s_min = cv2.getTrackbarPos("Sat Min", TrackBars)
  s_max = cv2.getTrackbarPos("Sat Max", TrackBars)
  v_min = cv2.getTrackbarPos("Val Min", TrackBars)
  v_max = cv2.getTrackbarPos("Val Max", TrackBars)
  # print(h_min, h_max, s_min, s_max, v_min, v_max)
  lower = np.array([h_min, s_min, v_min])
  upper = np.array([h_max, s_max, v_max])
  mask = cv2.inRange(imgHSV, lower, upper)
  imgResult = cv2.bitwise_and(img, img, mask=mask)
  return mask, imgResult


# optional mapping of values with morphological shapes
def morph_shape(val):
    if val == 0:
        return cv2.MORPH_RECT
    elif val == 1:
        return cv2.MORPH_CROSS
    elif val == 2:
        return cv2.MORPH_ELLIPSE
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('D:\It_Jim\Internship2021testtask.mp4')
# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('D:\It_Jim\\task1_res_mp4.avi', cv2.VideoWriter_fourcc(*'XVID'), 5, (frame_width, frame_height))

pause = False
# Read until video is completed
ret, frame = cap.read()
while(cap.isOpened()):
  # Capture frame-by-frame
  if(pause == False):
    ret, frame = cap.read()
  else:
    ret = True

  if ret == True:
    img_contour = frame.copy()
    # Display the resulting frame
    gauss = cv2.GaussianBlur(frame, (5, 5), 50)
    mask, imgResult = сolor_detection(gauss)
    Canny_min = cv2.getTrackbarPos("Canny Min", TrackBars)
    Canny_max = cv2.getTrackbarPos("Canny Max", TrackBars)
    edges = cv2.Canny(mask, Canny_min, Canny_max)

    dialation_kernal = cv2.getTrackbarPos("dialation kernal", TrackBars)
    dialation_iterations = cv2.getTrackbarPos("dialation iterations", TrackBars)
    kernel = np.ones((dialation_kernal, dialation_kernal), np.uint8)

    dialation = cv2.dilate(edges, kernel, iterations = dialation_iterations)


    getCountours(edges, img_contour)
    imgStack = stackImages(0.25, ([frame, gauss, imgResult],[mask, dialation, img_contour]))

    out.write(img_contour)
    cv2.imshow("Result", imgStack)
    cv2.imshow("Result contur", img_contour)


    # Press Q on keyboard to  exit
    if cv2.waitKey(1) & 0xFF == ord('p'):
      print("pause")
      pause = not pause
    # Press Q on keyboard to  exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # Break the loop
  else:
    cap = cv2.VideoCapture('D:\It_Jim\Internship2021testtask.mp4')

# When everything done, release the video capture object
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()