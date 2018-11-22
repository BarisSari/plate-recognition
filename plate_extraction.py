import cv2
import numpy as np
import PossibleChar
import PossiblePlate
import math
# import pytesseract

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


def maximize_contrast(gray):
    el = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    se_top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, el)
    se_black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, el)

    top_hat = cv2.add(gray, se_top_hat)
    final_image = cv2.subtract(top_hat, se_black_hat)

    return final_image


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = maximize_contrast(gray)
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    thresholded = cv2.adaptiveThreshold(blurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                        ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

    return gray, thresholded


def find_chars(image):
    chars = []  # this will be the return value
    copied = image.copy()
    # _, contours, _ = cv2.findContours(copied, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    _, contours, _ = cv2.findContours(copied, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for i in range(0, len(contours)):  # for each contour
        possible_char = PossibleChar.PossibleChar(contours[i])

        if (possible_char.intBoundingRectArea > MIN_PIXEL_AREA and
                possible_char.intBoundingRectWidth > MIN_PIXEL_WIDTH and
                possible_char.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
                MIN_ASPECT_RATIO < possible_char.fltAspectRatio < MAX_ASPECT_RATIO):
            chars.append(possible_char)

    return chars


def find_distance(first, second):
    return math.sqrt(((first.intCenterX - second.intCenterX) ** 2) + ((first.intCenterY - second.intCenterY) ** 2))


def find_angle(firstChar, secondChar):
    if float(abs(firstChar.intCenterX - secondChar.intCenterX)) != 0.0:
        angle = math.atan(float(abs(firstChar.intCenterY - secondChar.intCenterY))
                          / float(abs(firstChar.intCenterX - secondChar.intCenterX)))
    else:
        angle = 1.5708

    angle = angle * (180.0 / math.pi)

    return angle


def find_possible_matched_chars(possible_char, char_list):
    chars = []

    for char in char_list:
        if char == possible_char:
            continue

        distance = find_distance(possible_char, char)
        angle = find_angle(possible_char, char)
        area_diff = float(
            abs(char.intBoundingRectArea - possible_char.intBoundingRectArea)) / float(
            possible_char.intBoundingRectArea)

        width_diff = float(
            abs(char.intBoundingRectWidth - possible_char.intBoundingRectWidth)) / float(
            possible_char.intBoundingRectWidth)
        height_diff = float(
            abs(char.intBoundingRectHeight - possible_char.intBoundingRectHeight)) / float(
            possible_char.intBoundingRectHeight)

        if (distance < (possible_char.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
                angle < MAX_ANGLE_BETWEEN_CHARS and
                area_diff < MAX_CHANGE_IN_AREA and
                width_diff < MAX_CHANGE_IN_WIDTH and
                height_diff < MAX_CHANGE_IN_HEIGHT):
            chars.append(char)

    return chars


def find_matched_chars(possible_chars_list):
    chars = []

    for char in possible_chars_list:  # for each possible char in the one big list of chars
        possibly_matched_chars = find_possible_matched_chars(char, possible_chars_list)
        possibly_matched_chars.append(char)

        if len(possibly_matched_chars) < 3:
            continue

        # if we get here, the current list passed test as a "group" or "cluster" of matching chars
        chars.append(possibly_matched_chars)  # so add to our list of lists of matching chars
        new_possible_chars_list = list(set(possible_chars_list) - set(possibly_matched_chars))

        recursive_possible_chars_list = find_matched_chars(new_possible_chars_list)

        for possible_char in recursive_possible_chars_list:
            chars.append(possible_char)

        break

    return chars


def extract_plate(img, matched_chars):
    possible_plate = PossiblePlate.PossiblePlate()

    matched_chars.sort(key=lambda x: x.intCenterX)

    # calculate the center point of the plate
    cX = (matched_chars[0].intCenterX + matched_chars[
        len(matched_chars) - 1].intCenterX) / 2.0
    cY = (matched_chars[0].intCenterY + matched_chars[
        len(matched_chars) - 1].intCenterY) / 2.0

    center = cX, cY

    # calculate plate width and height
    plate_width = int((matched_chars[len(matched_chars) - 1].intBoundingRectX + matched_chars[
        len(matched_chars) - 1].intBoundingRectWidth - matched_chars[
                             0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    intTotalOfCharHeights = 0

    for matchingChar in matched_chars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight

    fltAverageCharHeight = intTotalOfCharHeights / len(matched_chars)

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

    # calculate correction angle of plate region
    fltOpposite = matched_chars[len(matched_chars) - 1].intCenterY - matched_chars[0].intCenterY
    fltHypotenuse = find_distance(matched_chars[0],
                                         matched_chars[len(matched_chars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

    # pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
    possible_plate.rrLocationOfPlateInScene = (tuple(center), (plate_width, intPlateHeight), fltCorrectionAngleInDeg)

    # get the rotation matrix for our calculated correction angle
    matrix = cv2.getRotationMatrix2D(tuple(center), fltCorrectionAngleInDeg, 1.0)
    height, width = img.shape[:2]  # unpack original image width and height
    rotated = cv2.warpAffine(img, matrix, (width, height))  # rotate the entire image
    cropped = cv2.getRectSubPix(rotated, (plate_width, intPlateHeight), tuple(center))

    # copy the cropped plate image into the applicable member variable of the possible plate
    possible_plate.imgPlate = cropped

    return possible_plate


def detect_possible_plates(img):
    plates = []

    cv2.destroyAllWindows()
    g, t = preprocess(img)
    chars = find_chars(t)

    # draw_contours
    # contours = []
    # height, width = img.shape[0], img.shape[1]
    # imgContours = np.zeros((height, width, 3), np.uint8)
    # for char in chars:
    #     contours.append(char.contour)
    # cv2.drawContours(imgContours, contours, -1, (255.0, 255.0, 255.0))
    # cv2.imshow("2b", imgContours)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    matched_chars = find_matched_chars(chars)

    # contours = []
    # imgContours = np.zeros((height, width, 3), np.uint8)
    # for listOfMatchingChars in matched_chars:
    #     color = np.uint8(np.random.uniform(0, 255, 3))
    #     c = tuple(map(int, color))
    #     for matchingChar in listOfMatchingChars:
    #         contours.append(matchingChar.contour)
    #     cv2.drawContours(imgContours, contours, -1, color=c)
    # cv2.imshow("3", imgContours)

    for char in matched_chars:  # for each group of matching chars
        possible_plate = extract_plate(img, char)  # attempt to extract plate

        if possible_plate.imgPlate is not None:  # if plate was found
            plates.append(possible_plate)  # add to list of possible plates

    # imgContours = np.zeros((height, width, 3), np.uint8)
    # for i in range(0, len(plates)):
    #     p2fRectPoints = cv2.boxPoints(plates[i].rrLocationOfPlateInScene)
    #     cv2.line(imgContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), (0.0, 0.0, 255.0), 2)
    #     cv2.line(imgContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), (0.0, 0.0, 255.0), 2)
    #     cv2.line(imgContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), (0.0, 0.0, 255.0), 2)
    #     cv2.line(imgContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), (0.0, 0.0, 255.0), 2)
        # cv2.imshow("4a", imgContours)

    return plates


'''
def findPossibleCharsInPlate(imgThresh):
    listOfPossibleChars = []
    imgThreshCopy = imgThresh.copy()
    _, contours, _ = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        possibleChar = PossibleChar.PossibleChar(contour)

        if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
                possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and
                possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
                MIN_ASPECT_RATIO < possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
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
        # asagidaki 2 fonksiyondan biri patliyor
        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale)
        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)
        if len(listOfListsOfMatchingCharsInPlate) == 0:
            intPlateCounter += 1
            possiblePlate.strChars = ""
            continue

        for i in range(len(listOfListsOfMatchingCharsInPlate)):
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
'''

for i in range(2, 3):
    file = 'dataset/' + str(i) + '.png'
    img = cv2.imread(file)
    possible_plates = detect_possible_plates(img)
    print('*'*95, "\nFor {}:".format(file))
    for plate in possible_plates:
        cv2.imshow('a', plate.imgPlate)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    #     gray = cv2.cvtColor(plate.imgPlate, cv2.COLOR_BGR2GRAY)
    #     chars = pytesseract.image_to_string(plate.imgPlate)
    #     chars = ''.join(c for c in chars if c.isalnum())
        # print(chars)
    # listOfPossiblePlates = detectCharsInPlates(plates)  # detect chars in plates
    # for i in range(len(listOfPossiblePlates)):
    #     print(plate.strChars)
    #     cv2.imwrite('{}.png'.format(i), listOfPossiblePlates[i].imgPlate)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # if len(listOfPossiblePlates) == 0:
    #     print("\nno license plates were detected\n")
    # else:
    #     listOfPossiblePlates.sort(key=lambda x: len(x.strChars), reverse=True)
    #     licPlate = listOfPossiblePlates[0]
    #     cv2.imshow("imgPlate", licPlate.imgPlate)
        # cv2.imshow("imgThresh", licPlate.imgThresh)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # chars = pytesseract.image_to_string(licPlate.imgThresh)
        # print(chars)