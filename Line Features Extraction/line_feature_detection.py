import cv2
import numpy as np


def eliminate_noise(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    a = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    return a


def detect_lines(img, ero):
    has_horizontal_line = False
    has_vertical_line = False

    # Detect edges
    edges = cv2.Canny(ero, 100, 150, apertureSize=3)

    # Detect lines based on the edges
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10
    )

    # Loop through each line and calculate the slope
    try:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1)

            # Determine if the line is horizontal or vertical
            if abs(slope) < 1:
                cv2.line(
                    img, (x1, y1), (x2, y2), (0, 255, 0), 2
                )  # Horizontal line (green)
                has_horizontal_line = True
            else:
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Vertical line (red)
                has_vertical_line = True
    except TypeError:
        print("No line is detected in the picture.")
        return

    if has_horizontal_line:
        print("This pic has horizontal line.")
    if has_vertical_line:
        print("This pic has vertical line.")


def detect_color(img):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the range of white color in HSV
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])

    # Create a mask for white color
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Define the range of yellow color in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    # Create a mask for yellow color
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    if np.sum(yellow_mask) > np.sum(white_mask):
        print("This pic has yellow line.")
    else:
        print("This pic has white line.")


# Load the image
img = cv2.imread("road-2.jpg")
# dark = cv2.convertScaleAbs(img, beta=-75)

res = eliminate_noise(img)
# res = eliminate_noise(dark)

detect_lines(img, res)
detect_color(img)

# Display the image
cv2.imshow("Image", img)
# cv2.imshow("Dark", dark)
cv2.imshow("Gray", res)
cv2.waitKey(0)
