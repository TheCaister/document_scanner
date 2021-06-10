import cv2.cv2 as cv2
import numpy as np

widthImg = 480
heightImg = 640

# Displaying a webcam
cap = cv2.VideoCapture(0)
# In set(), the first parameter is the ID of the property to change
# In this case, we're setting the width, height and brightness of the image
cap.set(3, widthImg)
cap.set(4, heightImg)
cap.set(10, 150)


def getContours(img):
    # Variables for holding properties of the largest contour
    biggest = np.array([])
    maxArea = 0

    # Use the findContours function, passing in RETR_EXTERNAL which returns the extreme outer contours
    # Passing in CHAIN_APPROX_NONE means we'll get ALL the contours. No compressed values
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Iterate through every contour and find their areas
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Make sure the areas are above a certain value to reduce noise
        if area > 5000:
            # -1 for contour index means that we want to draw EVERY contour
            # cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)

            # Getting the perimeters of every contour, passing in True because they're all closed
            perimeter = cv2.arcLength(cnt, True)

            # Create a rough polygon based on the contours
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

            # Find the biggest rectangular contour as we are iterating
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area

    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    # Return a list of points for the biggest contour
    return biggest


# Function for processing images and returning an image with edges
def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThreshold = cv2.erode(imgDil, kernel, iterations=1)

    return imgThreshold


# Function for reordering points so that the getWarp function works properly
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)

    # Get the sum of x's and y's for every point
    add = myPoints.sum(1)

    # First point is the smallest value in add
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]

    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew

# Function for getting warp properties
def getWarp(img, biggest):
    biggest = reorder(biggest)

    # Get points of the card and determine which corner they are
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    # Cropping the output image
    imgCropped = imgOutput[20:imgOutput.shape[0] - 20, 20:imgOutput.shape[1] - 20]

    return imgOutput


# Infinite loop for displaying frames captured by webcam
while True:
    success, img = cap.read()
    img = cv2.resize(img, (widthImg, heightImg))
    imgContour = img.copy()

    imgThres = preProcessing(img)
    biggest = getContours(imgThres)
    print(biggest)
    imgWarped = getWarp(img, biggest)
    # Rotating the webcam for my particular device
    cv2.imshow("Video", cv2.rotate(imgWarped, cv2.ROTATE_90_CLOCKWISE))
    # If Q is pressed, quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
