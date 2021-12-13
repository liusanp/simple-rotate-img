from rotate_img_test import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import cv2
import imutils
from rotate_img import rotate_bound, azimuth_angle, cal_angle, get_length


def test():
	# construct the argument parser and parse the arguments
	# ap = argparse.ArgumentParser()
	# ap.add_argument("-i", "--image", required = True,
	# 	help = "Path to the image to be scanned")
	# args = vars(ap.parse_args())

	# load the image and compute the ratio of the old height
	# to the new height, clone it, and resize it
	image = cv2.imread('C:\\Users\\liusa\\Desktop\\202199-95311.png')
	# image = cv2.imread('D:\\PycharmProjects\\vgg16\\data\\up\\202101203cy1bq.jpg')
	ratio = image.shape[0] / 500.0
	orig = image.copy()
	img_height = image.shape[0]
	img_width = image.shape[1]
	if img_width > img_height and img_width > 500:
		image = imutils.resize(image, width=500)
	elif img_height > img_width and img_height > 500:
		image = imutils.resize(image, height=500)
	line_img = image.copy()
	rotate_img = image.copy()
	# convert the image to grayscale, blur it, and find  edges
	# in the image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 100, 200)
	# edged = cv2.Canny(gray, 50, 150)

	minLineLength = 100
	maxLineGap = 5
	font = cv2.FONT_HERSHEY_SIMPLEX
	lines = cv2.HoughLinesP(edged, 1, np.pi/180, 100, minLineLength, maxLineGap)
	# print(lines)
	angles = []
	if lines is not None and lines.any():
		for l in lines:
			x1, y1, x2, y2 = l[0]
			if get_length(x1, y1, x2, y2) > 50:
				angle = azimuth_angle(x1, y1, x2, y2)
				if angle > 0 and angle < 90:
					angles.append(angle if angle > 0 else angle + 180)
				cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
				cv2.putText(line_img, str(angle), (x2, y2), font, 0.5, (255, 0, 0), 1)
		if len(angles) > 0:
			final_angle = cal_angle(angles)
			rotate_img = rotate_bound(rotate_img, final_angle if final_angle <= 45 else final_angle - 90)

	# show the original image and the edge detected image
	print("STEP 1: Edge Detection")
	cv2.imshow("Image", image)
	cv2.imshow("Edged", edged)
	cv2.imshow("Lines", line_img)
	cv2.imshow("rotate", rotate_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# find the contours in the edged image, keeping only the
	# largest ones, and initialize the screen contour
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
	# loop over the contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.05 * peri, True)
		# if our approximated contour has four points, then we
		# can assume that we have found our screen
		if len(approx) == 4:
			screenCnt = approx
			break
	# show the contour (outline) of the piece of paper
	print("STEP 2: Find contours of paper")
	cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
	cv2.imshow("Outline", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# apply the four point transform to obtain a top-down
	# view of the original image
	warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
	# convert the warped image to grayscale, then threshold it
	# to give it that 'black and white' paper effect
	warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	T = threshold_local(warped, 11, offset=10, method="gaussian")
	warped = (warped > T).astype("uint8") * 255
	# show the original and scanned images
	print("STEP 3: Apply perspective transform")
	cv2.imshow("Original", imutils.resize(orig, height=650))
	cv2.imshow("Scanned", imutils.resize(warped, height=650))
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	test()
	pass
