import cv2
import time
import handtrackingmodule as htm
import numpy as np
import os

overlayList = []  # list to store all the images

brushThickness = 25
eraserThickness = 100
drawColor = (82, 113, 255)  # setting red color

xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)  # defining canvas

# images in header folder
folderPath = "Header"
myList = os.listdir(folderPath)  # getting all the images used in code
for imPath in myList:  # reading all the images from the folder
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)  # inserting images one by one in the overlayList
    # print(len(overlayList))  # printing the length of the list
header = overlayList[0]  # storing 1st image
cap = cv2.VideoCapture(0)  # capturing video from webcam
cap.set(3, 1280)  # setting width
cap.set(4, 720)  # setting height

detector = htm.handDetector(detectionCon=0.50, maxHands=1)  # making object


while True:

    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)  # for neglecting mirror inversion

    # 2. Find Hand Landmarks
    img = detector.findHands(img)  # using functions fo connecting landmarks
    lmList, bbox = detector.findPosition(img, draw=False)  # using function to find specific landmark position, draw false means no circles on landmarks

    if len(lmList) != 0:
        # print(lmList)
        x1, y1 = lmList[8][1], lmList[8][2]  # tip of index finger
        x2, y2 = lmList[12][1], lmList[12][2]  # tip of middle finger

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        # 4. If Selection Mode - Two fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            # print("Selection Mode")
            # checking for click
            if y1 < 125:
                if 250 < x1 < 495:  # if I am clicking at purple brush
                    header = overlayList[0]
                    drawColor = (82, 113, 255) # setting red color
                elif 575 < x1 < 655:  # if I am clicking at green brush
                    header = overlayList[1]
                    drawColor = (0, 191, 99) # setting green color
                elif 750 < x1 < 835:  # if I am clicking at blue brush 
                    header = overlayList[2]
                    drawColor = (255, 49, 49) # setting blue color
                elif 875 < x1 < 900:  # if I am clicking at purple brush 
                    header = overlayList[3]
                    drawColor = (216, 58, 249) # setting purple color
                elif 1070 < x1 < 1200:  # if I am clicking at eraser
                    header = overlayList[4]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)  # selection mode is represented as a rectangle

        # 5. If Drawing Mode - Index finger is up
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)  # drawing mode is represented as a circle
            # print("Drawing Mode")
            if xp == 0 and yp == 0:  # initially xp and yp will be at 0,0 so it will draw a line from 0,0 to whichever point our tip is at
                xp, yp = x1, y1  # so to avoid that we set xp=x1 and yp=y1
            # till now we are creating our drawing but it gets removed as every time our frames are updating so we have to define our canvas where we can draw and show also

            # eraser
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)  # gonna draw lines from previous coordinates to new positions
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1  # giving values to xp,yp every time

    # 1 converting img to gray
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)

    # 2 converting into a binary image and then inverting
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)  # on canvas all the region in which we drew is black and where it is black it is considered as white, it will create a mask

    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)  # converting again to gray because we have to add in an RGB image i.e img

    # add the original img with imgInv, by doing this we get our drawing only in black color
    img = cv2.bitwise_and(img, imgInv)

    # add img and imgcanvas, by doing this we get colors on img
    img = cv2.bitwise_or(img, imgCanvas)

    # setting the header image
    header = cv2.resize(header, (1280, 140))
    img[0:140, 0:1280] = header # on our frame we are setting our JPG image according to H,W of jpg images

    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", imgCanvas)
    # cv2.imshow("Inv", imgInv)

    # 6. Quit the program when 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the camera capture
cap.release()
cv2.destroyAllWindows()
