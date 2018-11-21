import cv2
import numpy as np
import PossibleChar
import PossiblePlate
import math

kNearest = cv2.ml.KNearest_create()
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9
MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8
MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0
MIN_PIXEL_AREA = 80
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0
MAX_CHANGE_IN_AREA = 0.5
MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2
MAX_ANGLE_BETWEEN_CHARS = 12.0
PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5
RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30


def preprocess(image):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, _, gray = cv2.split(img_hsv)

    gray = maximize_contrast(gray)
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)

    thresholded = cv2.adaptiveThreshold(blurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                        ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

    return gray, thresholded


def maximize_contrast(gray):
    el = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, el)
    imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, el)
    imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    return imgGrayscalePlusTopHatMinusBlackHat


def checkIfPossibleChar(possibleChar):
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
            possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and
            possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
            MIN_ASPECT_RATIO < possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False


def find_chars(image):
    height, width = image.shape
    chars = []  # this will be the return value
    copied = image.copy()
    _, contours, _ = cv2.findContours(copied, cv2.RETR_LIST,
                                      cv2.CHAIN_APPROX_SIMPLE)  # find all contours

    img_contours = np.zeros((height, width, 3), np.uint8)

    for i in range(0, len(contours)):  # for each contour

        cv2.drawContours(img_contours, contours, i, (255.0, 255.0, 255.0))
        possibleChar = PossibleChar.PossibleChar(contours[i])

        if checkIfPossibleChar(possibleChar):
            chars.append(possibleChar)

    return chars


def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))


def angleBetweenChars(firstChar, secondChar):
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))

    if fltAdj != 0.0:  # check to make sure we do not divide by zero if the center X positions are equal, float division by zero will cause a crash in Python
        fltAngleInRad = math.atan(fltOpp / fltAdj)  # if adjacent is not zero, calculate angle
    else:
        fltAngleInRad = 1.5708  # if adjacent is zero, use this as the angle, this is to be consistent with the C++ version of this program
    # end if

    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)  # calculate angle in degrees

    return fltAngleInDeg


def findListOfMatchingChars(possibleChar, listOfChars):
    listOfMatchingChars = []  # this will be the return value

    for possibleMatchingChar in listOfChars:
        if possibleMatchingChar == possibleChar:
            continue

        fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)
        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)
        fltChangeInArea = float(
            abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(
            possibleChar.intBoundingRectArea)

        fltChangeInWidth = float(
            abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(
            possibleChar.intBoundingRectWidth)
        fltChangeInHeight = float(
            abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(
            possibleChar.intBoundingRectHeight)

        # check if chars match
        if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
                fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
                fltChangeInArea < MAX_CHANGE_IN_AREA and
                fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
                fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):
            listOfMatchingChars.append(possibleMatchingChar)

    return listOfMatchingChars


def findListOfListsOfMatchingChars(listOfPossibleChars):
    listOfListsOfMatchingChars = []

    for possibleChar in listOfPossibleChars:  # for each possible char in the one big list of chars
        listOfMatchingChars = findListOfMatchingChars(possibleChar, listOfPossibleChars)
        listOfMatchingChars.append(possibleChar)

        if len(listOfMatchingChars) < 3:
            continue

        # if we get here, the current list passed test as a "group" or "cluster" of matching chars
        listOfListsOfMatchingChars.append(listOfMatchingChars)  # so add to our list of lists of matching chars
        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))

        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(
            listOfPossibleCharsWithCurrentMatchesRemoved)

        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)

        break

    return listOfListsOfMatchingChars


def extractPlate(imgOriginal, listOfMatchingChars):
    possiblePlate = PossiblePlate.PossiblePlate()  # this will be the return value

    listOfMatchingChars.sort(
        key=lambda matchingChar: matchingChar.intCenterX)  # sort chars from left to right based on x position

    # calculate the center point of the plate
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[
        len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[
        len(listOfMatchingChars) - 1].intCenterY) / 2.0

    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

    # calculate plate width and height
    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[
        len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[
                             0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    intTotalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight

    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

    # calculate correction angle of plate region
    fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = distanceBetweenChars(listOfMatchingChars[0],
                                         listOfMatchingChars[len(listOfMatchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

    # pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
    possiblePlate.rrLocationOfPlateInScene = (
        tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg)

    # get the rotation matrix for our calculated correction angle
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)

    height, width, numChannels = imgOriginal.shape  # unpack original image width and height

    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))  # rotate the entire image

    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))

    # copy the cropped plate image into the applicable member variable of the possible plate
    possiblePlate.imgPlate = imgCropped

    return possiblePlate


def detectPlatesInScene(img):
    plates = []
    contours = []
    height, width = img.shape[0], img.shape[1]
    imgContours = np.zeros((height, width, 3), np.uint8)

    cv2.destroyAllWindows()
    g, t = preprocess(img)
    chars = find_chars(t)

    for char in chars:
        contours.append(char.contour)

    cv2.drawContours(imgContours, contours, -1, (255.0, 255.0, 255.0))
    cv2.imshow("2b", imgContours)

    listOfListsOfMatchingCharsInScene = findListOfListsOfMatchingChars(chars)

    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
        color = np.uint8(np.random.uniform(0, 255, 3))
        c = tuple(map(int, color))

        contours = []
        for matchingChar in listOfMatchingChars:
            contours.append(matchingChar.contour)

        cv2.drawContours(imgContours, contours, -1, color=c)

    cv2.imshow("3", imgContours)

    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:  # for each group of matching chars
        possiblePlate = extractPlate(img, listOfMatchingChars)  # attempt to extract plate

        if possiblePlate.imgPlate is not None:  # if plate was found
            plates.append(possiblePlate)  # add to list of possible plates

    for i in range(0, len(plates)):
        p2fRectPoints = cv2.boxPoints(plates[i].rrLocationOfPlateInScene)

        cv2.line(imgContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), (0.0, 0.0, 255.0), 2)
        cv2.line(imgContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), (0.0, 0.0, 255.0), 2)
        cv2.line(imgContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), (0.0, 0.0, 255.0), 2)
        cv2.line(imgContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), (0.0, 0.0, 255.0), 2)

        cv2.imshow("4a", imgContours)

    return plates


def findPossibleCharsInPlate(imgThresh):
    listOfPossibleChars = []
    imgThreshCopy = imgThresh.copy()
    _, contours, _ = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        possibleChar = PossibleChar.PossibleChar(contour)

        if checkIfPossibleChar(possibleChar):
            listOfPossibleChars.append(possibleChar)

    return listOfPossibleChars


def removeInnerOverlappingChars(listOfMatchingChars):
    listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)  # this will be the return value

    for currentChar in listOfMatchingChars:
        for otherChar in listOfMatchingChars:
            if currentChar != otherChar:  # if current char and other char are not the same char . . .
                # if current char and other char have center points at almost the same location . . .
                if distanceBetweenChars(currentChar, otherChar) < (
                        currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                    # if we get in here we have found overlapping chars
                    # next we identify which char is smaller, then if that char was not already removed on a previous pass, remove it
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:  # if current char is smaller than other char
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved:  # if current char was not already removed on a previous pass . . .
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)  # then remove current char
                        # end if
                    else:  # else if other char is smaller than current char
                        if otherChar in listOfMatchingCharsWithInnerCharRemoved:  # if other char was not already removed on a previous pass . . .
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)  # then remove other char

    return listOfMatchingCharsWithInnerCharRemoved


def recognizeCharsInPlate(imgThresh, listOfMatchingChars):
    strChars = ""

    height, width = imgThresh.shape
    imgThreshColor = np.zeros((height, width, 3), np.uint8)
    listOfMatchingChars.sort(key=lambda matchingChar: matchingChar.intCenterX)
    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor)

    for currentChar in listOfMatchingChars:
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth),
               (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))

        cv2.rectangle(imgThreshColor, pt1, pt2, (0.0, 0.0, 255.0), 2)
        imgROI = imgThresh[
                 currentChar.intBoundingRectY: currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                 currentChar.intBoundingRectX: currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]

        imgROIResized = cv2.resize(imgROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))
        npaROIResized = imgROIResized.reshape((1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))
        npaROIResized = np.float32(npaROIResized)
        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k=1)
        strCurrentChar = str(chr(int(npaResults[0][0])))
        strChars = strChars + strCurrentChar

    return strChars


def detectCharsInPlates(listOfPossiblePlates):
    intPlateCounter = 0

    if len(listOfPossiblePlates) == 0:
        return listOfPossiblePlates

    for possiblePlate in listOfPossiblePlates:
        possiblePlate.imgGrayscale, possiblePlate.imgThresh = preprocess(possiblePlate.imgPlate)

        possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx=1.6, fy=1.6)
        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0,
                                                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale)
        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)
        if (len(listOfListsOfMatchingCharsInPlate) == 0):
            intPlateCounter += 1
            possiblePlate.strChars = ""
            continue

        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            listOfListsOfMatchingCharsInPlate[i].sort(key=lambda matchingChar: matchingChar.intCenterX)
            listOfListsOfMatchingCharsInPlate[i] = removeInnerOverlappingChars(listOfListsOfMatchingCharsInPlate[i])

        intLenOfLongestListOfChars = 0
        intIndexOfLongestListOfChars = 0

        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
                intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
                intIndexOfLongestListOfChars = i

        longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]
        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThresh, longestListOfMatchingCharsInPlate)

    return listOfPossiblePlates


for i in range(2, 5):
    file = 'dataset/' + str(i) + '.png'
    img = cv2.imread(file)
    plates = detectPlatesInScene(img)
    for plate in plates:
        cv2.imshow('a', plate.imgPlate)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    # listOfPossiblePlates = detectCharsInPlates(plates)  # detect chars in plates
    # if len(listOfPossiblePlates) == 0:
    #     print("\nno license plates were detected\n")
    # else:
    #     listOfPossiblePlates.sort(key=lambda x: len(x.strChars), reverse=True)
    #     licPlate = listOfPossiblePlates[0]
    #     cv2.imshow("imgPlate", licPlate.imgPlate)
    #     cv2.imshow("imgThresh", licPlate.imgThresh)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
