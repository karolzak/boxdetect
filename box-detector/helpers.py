import cv2
import numpy as np
import imutils


def apply_thresholding(image):
    otsu  = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    binary = cv2.threshold(image, np.mean(image) , 255, cv2.THRESH_BINARY_INV)[1]
    image = otsu + binary
	# image = cv2.threshold(image, np.median(image), 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

	# image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
	#             cv2.THRESH_BINARY_INV,5,3)
	# image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
	#             cv2.THRESH_BINARY_INV,3,5)
    return image


def get_rect_kernels(
		wh_ratio_range = (0.5, 1.1),
		min_w = 40,
		max_w = 60,
		min_h = 40,
		max_h = 65,
		pad = 1
	):

	kernels = [
		np.pad(np.zeros((h,w), dtype=np.uint8), pad, constant_values=1)
		for w in range(min_w, max_w)
		for h in range(min_h, max_h)
		if w/h >= wh_ratio_range[0] and w/h <= wh_ratio_range[1]
	]

	return kernels


def get_line_kernels(length):
    kernels = [
		np.ones((length,1), dtype=np.uint8),
		np.ones((1,length), dtype=np.uint8),
		# np.pad(np.ones((3,2), dtype=np.uint8), ((0,0),(1,0)), constant_values=0),
		# np.pad(np.ones((3,2), dtype=np.uint8), ((0,0),(0,1)), constant_values=0),
		# np.pad(np.ones((2,3), dtype=np.uint8), ((1,0),(0,0)), constant_values=0),
		# np.pad(np.ones((2,3), dtype=np.uint8), ((0,1),(0,0)), constant_values=0),
    ]
    return kernels


def detect_shape(c):
	# initialize the shape name and approximate the contour
	shape = "unidentified"
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.04 * peri, True)

	# if the shape has 4 vertices, it is either a square or
	# a rectangle
	if len(approx) == 4:
		# compute the bounding box of the contour and use the
		# bounding box to compute the aspect ratio
		(x, y, w, h) = cv2.boundingRect(approx)
		ar = w / float(h)

		# a square will have an aspect ratio that is approximately
		# equal to one, otherwise, the shape is a rectangle
		shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

	# return the name of the shape
	return shape


def get_contours(image, image_org, area_range=(2000,3000), thickness=1):
	# find contours in the thresholded image
	cnts = cv2.findContours(
		image,
		cv2.RETR_LIST,
		cv2.CHAIN_APPROX_SIMPLE
	)
	cnts = imutils.grab_contours(cnts)

	# loop over the contours
	for c in cnts:
		area = cv2.contourArea(c)
		
		if area > area_range[0] and area < area_range[1]:
		# if area > 50:
			shape = detect_shape(c)
			# if shape == "unidentified":
			# 	continue
		else:
			continue

		cv2.drawContours(image_org, [c], -1, (0, 255, 0), thickness)
	return image_org