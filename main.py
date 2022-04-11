# Step 1 Import Map
import cv2
import sys
import numpy as np
import imutils
from skimage.morphology import medial_axis
from skimage import morphology


class color_data:
    def __init__(self, color):
        self.hsv = color
        self.hue = self.hsv[0]
        self.sat = self.hsv[1]
        self.val = self.hsv[2]
        if self.val <= 200:
            thresh_val = 1
            self.lower = np.array([0, 0, 130])
            self.upper = np.array([0, thresh_val, 130 + thresh_val])
        else:
            thresh = 9
            if self.hue >= thresh:
                self.lower = np.array([self.hue - thresh, 60, 60])
                self.upper = np.array([self.hue + thresh, 255, 255])
            else:
                self.lower = np.array([1, 60, 60])
                self.upper = np.array([self.hue + thresh, 255, 255])


"""
-------------------------------------------------------------------
    Color Code                RGB -> HSV
-------------------------------------------------------------------
    Dark Gray :  130, 130, 130    ->     0,   0, 130   [Buildings]
    Dark Blue :    0, 112, 255    ->    13, 255, 255   [ADA Paths]
    Light Blue:  115, 223, 225    ->    29, 125, 225   [ADA Doors]
-------------------------------------------------------------------
"""
font = cv2.FONT_HERSHEY_PLAIN
dark_gray = color_data([0, 0, 130])
dark_blue = color_data([107, 255, 255])
light_blue = color_data([91, 125, 225])

color_list = [dark_gray, dark_blue, light_blue]


def distance(x1, x2, y1, y2):
    return np.sqrt(np.power(x2 - x1, 2) + (np.power(y2 - y1, 2)))


def contoursConvexHull(contours):
    pts = []
    for i in range(0, len(contours)):
        for j in range(0, len(contours[i])):
            pts.append(contours[i][j])

    pts = np.array(pts)
    result = cv2.convexHull(pts)
    return result


def combine_neighbors(cnt1, cnt2):
    row1, row2 = cnt1.shape[0], cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i] - cnt2[j])
            if abs(dist) < 50:
                return True
            elif i == row1 - 1 and j == row2 - 1:
                return False


def main():
    # Load Disability Map in BGR color format
    image = cv2.imread("1.png")

    # Catch Loading Failure
    if image is None:
        sys.exit("Could not read image.")

    # Convert Image from BGR to HSV for ease of manipulation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Resize Image Constants
    size_x = 1400
    size_y = int(size_x * 0.75)

    # Acquire Outlines
    dark_gray_hsv_threshold = cv2.inRange(hsv_image, dark_gray.lower, dark_gray.upper)  # Buildings
    dark_blue_hsv_threshold = cv2.inRange(hsv_image, dark_blue.lower, dark_blue.upper)  # Paths
    light_blue_hsv_threshold = cv2.inRange(hsv_image, light_blue.lower, light_blue.upper)  # Entry Points

    # Resize for previews
    r_dark_gray_hsv_threshold = cv2.resize(dark_gray_hsv_threshold, (size_x, size_y))
    r_dark_blue_hsv_threshold = cv2.resize(dark_blue_hsv_threshold, (size_x, size_y))
    r_light_blue_hsv_threshold = cv2.resize(light_blue_hsv_threshold, (size_x, size_y))

    # Combine images
    threshold_window = cv2.hconcat([
        r_dark_gray_hsv_threshold,
        r_dark_blue_hsv_threshold,
        r_light_blue_hsv_threshold])

    # Dark Gray - Buildings ------------------------------------------------------

    kernel = np.ones((2, 2), np.uint8)
    dark_gray_result = cv2.bitwise_and(image, image, mask=dark_gray_hsv_threshold)
    dark_gray_result = cv2.morphologyEx(dark_gray_result, cv2.MORPH_OPEN, kernel, iterations=2)

    dgray_line = cv2.cvtColor(dark_gray_result, cv2.COLOR_BGR2GRAY)
    cont = cv2.findContours(dgray_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont = imutils.grab_contours(cont)

    for c in cont:
        epsilon = 0.001 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        cv2.drawContours(dgray_line, [approx], -1, (130, 0, 130), cv2.FILLED, lineType=cv2.LINE_AA)

    r_dark_gray_result = cv2.cvtColor(dgray_line, cv2.COLOR_GRAY2BGR)
    r_dark_gray_result = cv2.resize(r_dark_gray_result, (size_x, size_y))

    # Dark Blue - Paths ------------------------------------------------------
    dark_blue_result = cv2.bitwise_and(image, image, mask=dark_blue_hsv_threshold)
    dark_blue_result = cv2.morphologyEx(dark_blue_result, cv2.MORPH_OPEN, kernel, iterations=1)
    r_dark_blue_result = cv2.resize(dark_blue_result, (size_x, size_y))

    gray_line = cv2.cvtColor(dark_blue_result, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((1, 1), np.uint8)

    closing1 = cv2.morphologyEx(gray_line, cv2.MORPH_DILATE, kernel, iterations=1)
    closing2 = imutils.skeletonize(closing1, size=(5, 5))

    edges = closing1

    skelly = medial_axis(edges).astype(np.uint8)

    lines = cv2.HoughLinesP(edges, 3, np.pi / 180, 1, minLineLength=2, maxLineGap=5)
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            if distance(x1, y1, x2, y2) < 30:
                cv2.line(edges, (x1, y1), (x2, y2), (255, 0, 255), 1)

    #cv2.imshow('tte', closing2)

    # Light Blue - Entrances ------------------------------------------------------
    light_blue_result = cv2.bitwise_and(image, image, mask=light_blue_hsv_threshold)
    light_blue_result = cv2.morphologyEx(light_blue_result, cv2.MORPH_OPEN, kernel, iterations=1)
    r_light_blue_result = cv2.resize(light_blue_result, (size_x, size_y))

    circle_find = cv2.cvtColor(light_blue_result, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(circle_find, cv2.HOUGH_GRADIENT, dp=1, minDist=9,
                               param1=100, param2=5,
                               minRadius=3, maxRadius=5)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv2.circle(circle_find, center, 1, (255, 255, 255), 1)
        # circle outline
        radius = i[2]

        string = "   X: " + str(i[0]) + " Y: " + str(i[1])

        # String containing the co-ordinates.
        cv2.putText(circle_find, string, (i[0], i[1]), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 255), 1)
        cv2.circle(circle_find, center, radius, (255, 0, 0), 5)

    result_window = cv2.hconcat([r_dark_gray_result,
                                 r_dark_blue_result,
                                 r_light_blue_result])

    circle_find = cv2.cvtColor(circle_find, cv2.COLOR_GRAY2BGR)

    final = cv2.bitwise_or(circle_find, dark_blue_result, mask=None)
    final = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)

    _, threshold = cv2.threshold(final, 150, 255, cv2.THRESH_BINARY_INV)

    contours = cv2.findContours(final.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    edges = cv2.Canny(final, 100, 200, apertureSize=3)

    for cnt in contours:
        epsilon = 0.0001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(edges, [approx], -1, (255, 255, 255), 1)

    _, thres = cv2.threshold(edges, 150, 255, cv2.THRESH_OTSU)

    r_circle = cv2.resize(circle_find, (size_x, size_y))
    trial = cv2.bitwise_or(r_light_blue_result, r_dark_blue_result)
    trial = cv2.bitwise_or(trial, r_dark_gray_result)
    trial = cv2.bitwise_or(trial, r_circle)
    cv2.imshow("Processed", trial)



if __name__ == "__main__":
    main()
    cv2.waitKey()
