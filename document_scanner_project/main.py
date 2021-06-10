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

    # Cropping the output image by shaving off 20 pixels from every edge
    imgCropped = imgOutput[20:imgOutput.shape[0] - 20, 20:imgOutput.shape[1] - 20]
    imgCropped = cv2.resize(imgCropped, (widthImg, heightImg))

    return imgOutput


# Takes a value for scale and an array of images. Stacks them.
def stackImages(scale, imgArray):
    # Get the amount of rows and columns
    rows = len(imgArray)
    cols = len(imgArray[0])

    # Checks if imgArray[0] is an instance of a list
    # Basically, checks to see if we're working with a 1D array or 2D array of images.
    # We'll know it's a 2D array if the first element of imgArray is a list of images.
    # If it's a list, assign True to rowsAvailable
    rowsAvailable = isinstance(imgArray[0], list)

    # Get the second dimension of each element of imgArray[0]
    # If it's a 2D array, get the widths and heights of the first image.
    # Otherwise, if it's a 1D array, get the width and height of the first image's first row.
    # The values will be thrown out if it's a 1D array.
    width = imgArray[0][0].shape[1]
    # Get the first dimension of each element of imgArray[0]
    height = imgArray[0][0].shape[0]

    # If there are rows available, display the 2D array of images
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    # Otherwise, display all the images in the 1D list given.
    else:
        # For every image
        for x in range(0, rows):
            # If image x's dimensions match image 0's dimensions, resize image x using the scale given.
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            # Otherwise, resize image x to the size of image 0 before scaling it
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            # If image x has a length of 2(Only height and width, no colour),
            # convert it so that it has a colour channel thingy
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        # Once all the images are nicely converted, stack them.
        hor = np.hstack(imgArray)
        ver = hor
    return ver


# Infinite loop for displaying frames captured by webcam
while True:
    success, img = cap.read()
    img = cv2.resize(img, (widthImg, heightImg))
    imgContour = img.copy()

    imgThres = preProcessing(img)
    biggest = getContours(imgThres)

    # If there is a biggest, do what's necessary
    if biggest.size != 0:
        imgWarped = getWarp(img, biggest)
        imageArray = ([img, imgThres],
                      [imgContour, imgWarped])
    # Otherwise, just display the normal images
    else:
        imageArray = ([img, imgThres],
                      [img, img])

    stackedImages = stackImages(0.6, imageArray)

    # Rotating the webcam for my particular device
    cv2.imshow("Video", cv2.rotate(stackedImages, cv2.ROTATE_90_CLOCKWISE))
    # If Q is pressed, quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
