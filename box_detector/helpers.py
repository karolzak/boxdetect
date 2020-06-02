import cv2
import numpy as np
import imutils
import random


def enhance_rectangles(image, kernels, plot=False):
	# new_image = np.zeros_like(image)
	# for kernel in kernels:
	# 	morphs = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)
	# 	new_image += morphs
	# image = new_image

	new_image = np.zeros_like(image)
	for kernel in kernels:
		morphs = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
		new_image += morphs
	image = new_image
	image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)[1]

	if plot:
		cv2.imshow("rectangular shape enhanced image", image)
		cv2.waitKey(0)
	
	return image


def enhance_image(image, kernels, plot=False):
	new_image = np.zeros_like(image)
	for kernel in kernels:
		morphs = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)

		# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,1))
		# morphs = cv2.dilate(morphs, kernel, iterations = 1)

		# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,3))
		# morphs = cv2.dilate(morphs, kernel, iterations = 1)

		morphs = cv2.morphologyEx(morphs, cv2.MORPH_OPEN, kernel, iterations=1)
		new_image += morphs
	image = new_image
	image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)[1]

	if plot:
		cv2.imshow("enhanced image", image)
		cv2.waitKey(0)

	return image


def apply_thresholding(image, plot=False):
	otsu  = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
	binary = cv2.threshold(image, np.mean(image) , 255, cv2.THRESH_BINARY_INV)[1]
	image = otsu + binary
	# image = cv2.threshold(image, np.median(image), 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

	# image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
	#             cv2.THRESH_BINARY_INV,5,3)
	# image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
	#             cv2.THRESH_BINARY_INV,3,5)
	if plot:
		cv2.imshow("thresholded image", image)
		cv2.waitKey(0)
	return image


def get_rect_kernels(
		wh_ratio_range=(0.5, 1.1),
		min_w=40,
		max_w=60,
		min_h=40,
		max_h=65,
		pad=1
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
		np.ones((length, 2), dtype=np.uint8),
		np.ones((2, length), dtype=np.uint8),
		# np.pad(np.ones((3,2), dtype=np.uint8), ((0,0),(1,0)), constant_values=0),
		# np.pad(np.ones((3,2), dtype=np.uint8), ((0,0),(0,1)), constant_values=0),
		# np.pad(np.ones((2,3), dtype=np.uint8), ((1,0),(0,0)), constant_values=0),
		# np.pad(np.ones((2,3), dtype=np.uint8), ((0,1),(0,0)), constant_values=0),
    ]
    return kernels


def group_countours(cnts):
	rects = [get_bounding_rect(c)[:4] for c in cnts]
	# we need to duplicate all the rects for grouping below to work
	rects += rects
	rects, weights = cv2.groupRectangles(rects, 1, 0.2)
	return rects


def get_bounding_rect(c):
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 3, True)
	(x, y, w, h) = cv2.boundingRect(approx)
	if len(approx) == 4:
		return x, y, w, h, True
	return x, y, w, h, False


def check_rect_ratio(c, wh_ratio_range):
	(x, y, w, h, is_rect) = get_bounding_rect(c)
	ar = w / float(h)
	if is_rect and ar >= wh_ratio_range[0] and ar <= wh_ratio_range[1]:
		return True
	return False


def filter_contours_by_rect_ratio(cnts, wh_ratio_range):
	return [
		c for c in cnts
		if check_rect_ratio(c, wh_ratio_range)
	]


def get_contours(image):
	# find contours in the thresholded image
	cnts = cv2.findContours(
		image.copy(),
		cv2.RETR_LIST,
		cv2.CHAIN_APPROX_SIMPLE
	)
	cnts = imutils.grab_contours(cnts)
	return cnts


def filter_contours_by_area_size(cnts, area_range):
	cnts_filtered = []
	for c in cnts:
		area = cv2.contourArea(c)
		if area > area_range[0] and area < area_range[1]:
			cnts_filtered.append(c)
	return cnts_filtered


def rescale_contours(cnts, ratio):
	cnts_rescaled = []
	for c in cnts:
		c = c.astype("float")
		c *= ratio
		c = c.astype("int")
		cnts_rescaled.append(c)
	return cnts_rescaled


def get_grouping_rectangles(rect_groups):
    rectangles = []
    for group in rect_groups:
        points = []
        for rect in group:
            points.append((rect[0], rect[1]))
            points.append((rect[0] + rect[2], rect[1] + rect[3]))
        rectangles.append(cv2.boundingRect(np.asarray(points)))
    return rectangles


def draw_rects(image, rects, color=(0, 255, 0), thickness=1):
	# loop over the contours
	for r in rects:
		x, y, w, h = r
		# cv2.drawContours(image, [c], -1, (random.sample(range(0, 255), 3)), thickness)
		cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness)
	return image


def _create_vert_group(rects, y, padding):
    indices = np.where(
        np.isin(
            rects[:,1],
            list(range(y - padding // 2,
                       y + padding // 2, 1))
        ))
    if len(indices[0]) > 0:
        return indices[0].astype(np.int)
    else:
        return None
    

def group_rects_vertically(rects, padding=10, max_width=4000, min_group_size=2):
    assert(padding % 2 == 0)
    vertical_groups = [
        group for group in
        (_create_vert_group(rects, y, padding)
         for y in range(padding // 2, max_width, padding))
        if group is not None
    ]
    return [group for group in vertical_groups if len(group) >= min_group_size]


def get_horizontal_groups_from_vert_groups(
		rects, vertical_groups_indices,
		max_distance, min_group_size):
	rect_groups = [
		rect_group
		for rect_group in(
			group_rects_horizontally(rects, group_indices, max_distance=max_distance, min_group_size=min_group_size)
			for group_indices in vertical_groups_indices
		)
		if rect_group != []
	]
	rect_groups = [
		horizontal_group
		for vertical_group in rect_groups
		for horizontal_group in vertical_group
	]
	return rect_groups


def group_rects_horizontally(rects, vert_group_indices, max_distance, min_group_size=1):
    vert_group = np.take(rects, vert_group_indices, axis=0)
    vert_group_sorted = vert_group[np.argsort(vert_group[:, 0])]
    horizontal_groups = []
    temp_group = []
    temp_group.append(vert_group_sorted[0])
    for i in range(1, len(vert_group_sorted)):
        rect1 = vert_group_sorted[i-1]
        x1 = rect1[0] + rect1[2]
        
        rect2 = vert_group_sorted[i]
        x2 = rect2[0]
        
        distance = abs(x2 - x1)
        if distance <= max_distance:
            temp_group.append(rect2)
        else:
            horizontal_groups.append(temp_group)
            temp_group = []
            temp_group.append(rect2)
        
        if i == len(vert_group_sorted) - 1:
            horizontal_groups.append(temp_group)
    horizontal_groups = [group for group in horizontal_groups if len(group) >= min_group_size]
    return horizontal_groups


def group_rects(rects, max_distance, rect_indices=None, min_group_size=1, grouping_mode='vertical'):
	grouping_mode = grouping_mode.lower()
	assert(grouping_mode in ['vertical', 'horizontal'])
	if grouping_mode == 'vertical':
		m, n = (0, 2)
	elif grouping_mode == 'horizontal':
		m, n = (1, 3)

	if rect_indices:
		rects = np.take(rects, rect_indices, axis=0)	

	rects_sorted = rects[np.argsort(rects[:, m])]
	new_groups = []
	temp_group = []
	temp_group.append(rects_sorted[0])
	for i in range(1, len(rects_sorted)):
		rect1 = rects_sorted[i-1]
		x1 = rect1[m] + rect1[n]

		rect2 = rects_sorted[i]
		x2 = rect2[m]

		distance = abs(x2 - x1)
		if distance <= max_distance:
		    temp_group.append(rect2)
		else:
		    new_groups.append(temp_group)
		    temp_group = []
		    temp_group.append(rect2)

		if i == len(rects_sorted) - 1:
			new_groups.append(temp_group)
	new_groups = [group for group in new_groups if len(group) >= min_group_size]
	return new_groups
