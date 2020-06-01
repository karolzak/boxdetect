# USAGE
# python detect_boxes.py

# import the necessary packages
import argparse
import cv2
from PIL import Image
import numpy as np
import PIL.ImageOps  
import glob
import sys
sys.path.append('./box_detector')
from box_detector.helpers import (
	get_line_kernels, get_rect_kernels,
	get_contours, apply_thresholding
)


if __name__ == "__main__":		
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image_dir", required=True,
		help="path to the input dir")
	ap.add_argument("-p", "--plot", type=bool, default=False, required=False,
		help="plot results")
	args = vars(ap.parse_args())

	images = glob.glob(args["image_dir"] + '*')
	plot = bool(args["plot"])
	print(args["image_dir"], plot)

	for img_path in images[:1]:
		print(img_path)
		# load the image and resize it to a smaller factor so that
		# the shapes can be approximated better
		# image = Image.open(args['image']).convert('RGB')
		# image =  PIL.ImageOps.invert(image)
		# image_org = np.asarray(image)
		image_org = cv2.imread(img_path)
		# image_org = imutils.resize(image_org, width=image_org.shape[0]//2)

		image = image_org.copy()
		# convert the resized image to grayscale
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = apply_thresholding(image)
		if plot:
			cv2.imshow("thresh", image)
			cv2.waitKey(0)

		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
		image = cv2.dilate(image, kernel, iterations = 1)
		
		if plot:
			cv2.imshow("thresh", image)
			cv2.waitKey(0)

		kernels = get_line_kernels(length=5)

		new_image = np.zeros_like(image)
		for kernel in kernels:
			morphs = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2)

			kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,1))
			morphs = cv2.dilate(morphs, kernel, iterations = 1)

			kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,3))
			morphs = cv2.dilate(morphs, kernel, iterations = 1)

			morphs = cv2.morphologyEx(morphs, cv2.MORPH_OPEN, kernel, iterations=1)
			new_image += morphs		
		image = new_image			
		image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)[1]

		if plot:
			cv2.imshow("thresh", image)
			cv2.waitKey(0)
		
		kernels = get_rect_kernels(
			wh_ratio_range = (0.5, 1.0),
			min_w = 45,	max_w = 50,
			min_h = 55,	max_h = 60,
			pad=1)

		new_image = np.zeros_like(image)
		for kernel in kernels:
			morphs = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
			new_image += morphs			
		image = new_image
		image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)[1]

		if plot:
			cv2.imshow("thresh", image)
			cv2.waitKey(0)

		# find contours in the thresholded image and initialize the
		# shape detector
		image_org = get_contours(image, image_org, area_range=(2000,3000), thickness=2)
		if plot:
			cv2.imshow("Image", image_org)
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
