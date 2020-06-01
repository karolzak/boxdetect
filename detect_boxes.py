# USAGE
# python detect_boxes.py

# import the necessary packages
import argparse
import cv2
from PIL import Image
import numpy as np
import PIL.ImageOps  
import glob
import imutils

import sys
sys.path.append('./box-detector/')
from helpers import (
	get_line_kernels, get_rect_kernels,
	get_contours, apply_thresholding,
	enhance_image, enhance_rectangles,
	draw_contours, rescale_contours,
	filter_contours_by_area_size
)


if __name__ == "__main__":		
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--image_dir", required=True,
		help="path to the input dir")
	ap.add_argument("-p", "--plot", type=bool, default=False, required=False,
		help="plot results")
	ap.add_argument("-mp", "--multiprocessing", type=bool, default=False, required=False,
		help="use multiprocessing")
	args = vars(ap.parse_args())

	# parameters
	min_w, max_w = (40, 50)
	min_h, max_h = (50, 60)

	wh_ratio_range = (0.6, 1.0)
	padding = 1
	thickness = 2
	processing_time_width = 1000

	images = glob.glob(args["image_dir"] + '*')
	plot = bool(args["plot"])
	print(args["image_dir"], plot)

	for img_path in images[:]:
		print(img_path)
		# load the image and resize it to a smaller factor so that
		# the shapes can be approximated better
		# image = Image.open(args['image']).convert('RGB')
		# image =  PIL.ImageOps.invert(image)
		# image_org = np.asarray(image)
		image_org = cv2.imread(img_path)

		resize_ratio = image_org.shape[1] / processing_time_width
		resize_ratio_inv = processing_time_width / image_org.shape[1]

		min_w_res = int(min_w * resize_ratio_inv)
		max_w_res = int(max_w * resize_ratio_inv)
		min_h_res = int(min_h * resize_ratio_inv)
		max_h_res = int(max_h * resize_ratio_inv)

		area_range = (
			round(min_w_res * max_w_res * 0.80),
			round(max_w_res * max_h_res * 1.00)
		)		

		# resize the image for processing time
		image = image_org.copy()
		image = imutils.resize(image, width=processing_time_width)
		# image_org = imutils.resize(image_org, width=image_org.shape[0]//4)


		# convert the resized image to grayscale
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# apply tresholding to get all the pixel values to either 0 or 255
		# this function also inverts colors (black pixels will become the background)
		image = apply_thresholding(image, plot)

		# basic pixel inflation
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
		image = cv2.dilate(image, kernel, iterations = 1)
		if plot:
			cv2.imshow("thresh", image)
			cv2.waitKey(0)

		# creating line-shape kernels to be used for image enhancing step
		# try it out only in case of very poor results with previous setup
		# kernels = get_line_kernels(length=4)
		# image = enhance_image(image, kernels, plot)

		# creating rectangular-shape kernels to be used for extracting rectangular shapes		
		kernels = get_rect_kernels(
			wh_ratio_range = wh_ratio_range,
			min_w = min_w_res,	max_w = max_w_res,
			min_h = min_h_res,	max_h = max_h_res,
			pad=padding)
		image = enhance_rectangles(image, kernels, plot)

		# find contours in the thresholded image and initialize the
		# shape detector
		cnts = get_contours(image)
		cnts = filter_contours_by_area_size(cnts, area_range)
		cnts = rescale_contours(cnts, resize_ratio)
		image_org = draw_contours(image_org, cnts, thickness=thickness)
		if plot:
			cv2.imshow("Org image with boxes", image_org)
			cv2.waitKey(0)

		cv2.imwrite(img_path.replace('\\in','\\out'), image_org)

			# masked_image = cv2.addWeighted(
			# 	image_org, 1.0,
			# 	np.stack([
			# 		np.zeros_like(image), 
			# 		np.zeros_like(image),
			# 		image, #image, image
			# 	], axis=-1), 1, 0)

			# cv2.imshow('masked', masked_image)
			# cv2.imwrite(args['image'].replace('\\in','\\out'), masked_image)


			# kernel = np.ones((2,2),np.uint8)
			# image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
			# kernel = np.ones((2,1),np.uint8)
			# image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
			# image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)
			# kernel = np.ones((1,2),np.uint8)
			# image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
			# image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)
			# image = cv2.bitwise_not(image)
			# print(image.shape)
			# cv2.imwrite(args['image'].replace('\\in','\\out'), image)
			# cv2.imshow('masked', image)

			# cv2.imwrite(args['image'].replace('form','form_out'), image_org)
			# cv2.waitKey(0)
