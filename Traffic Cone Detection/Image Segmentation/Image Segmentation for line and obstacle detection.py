import cv2
import numpy as np

# Load the image
img = cv2.imread('maxresdefault.jpg')
img = cv2.resize(img, (500, 400))
# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blur, 50, 150)

# Create a region of interest mask
mask = np.zeros_like(edges)
height, width = img.shape[:2]
vertices = np.array([[(0, height), (width//2, height//2), (width, height)]], dtype=np.int32)
cv2.fillPoly(mask, vertices, 255)

# Apply the mask to the edge image
masked_edges = cv2.bitwise_and(edges, mask)

# Apply Hough transform
lines = cv2.HoughLinesP(masked_edges, rho=2, theta=np.pi/180, threshold=50, minLineLength=40, maxLineGap=100)

# Create a blank image to draw the lines and rectangles on
line_img = np.zeros((height, width, 3), dtype=np.uint8)
obj_img = np.zeros((height, width, 3), dtype=np.uint8)

# Draw the detected lines and rectangles on the blank images
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 5)

# Find contours of objects
contours, _ = cv2.findContours(masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw a rectangle around each object
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(obj_img, (x, y), (x+w, y+h), (0, 0, 255), 2)

# Merge the detected lines and object images with the original image
result = cv2.addWeighted(img, 0.8, line_img, 1, 0)
result = cv2.addWeighted(result, 0.8, obj_img, 1, 0)

# Display the final image
cv2.imshow('Original Picture', img)
cv2.imshow('Result', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
