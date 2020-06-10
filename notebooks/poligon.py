import cv2
import numpy as np
import sys
sys.path.append("../.")

from boxdetect import pipelines, config, img_proc


def DefaultConfig():
    config.width_range = (40, 55)
    config.height_range = (40, 55)
    config.scaling_factors = [1.0]
    config.wh_ratio_range = (0.8, 1.2)
    config.group_size_range = (1, 100)
    config.dilation_iterations = 0
    return config


file_path = "../tests/data/dummy_example.png"

results = pipelines.get_checkboxes(file_path, DefaultConfig(), plot=False)

print(results[:, 1])
print(np.sum(results[:, 1]))

for im in results[:, 2]:
    cv2.imshow("checkbox", im)
    cv2.waitKey(0)


# img = cv2.imread(file_path)
# # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# cfg = DefaultConfig()
# cfg.group_size_range = (1, 1)

# rects, grouping_rects, image, output_image = pipelines.get_boxes(
#     img, config=cfg, plot=True)

# pix_threshold = 0.1

# try:
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# except Exception as e:
#     print("Warning: failed to convert to grayscale...")
#     print(e)

# # apply tresholding to get all the pixel values to either 0 or 255
# # this function also inverts colors
# # (black pixels will become the background)
# img = img_proc.apply_thresholding(img, False)
# img = img_proc.draw_rects(
#     img, grouping_rects, color=(0, 0, 0), thickness=5)

# for rect in grouping_rects:
#     width = rect[2]
#     height = rect[3]
#     w_pad = int(width * 0.15)
#     h_pad = int(width * 0.15)

#     x1 = rect[0]
#     y1 = rect[1]
#     x2 = x1 + rect[2]
#     y2 = y1 + rect[3]
#     im_crop = img[y1+h_pad:y2-h_pad, x1+w_pad:x2-w_pad]
#     max_pix = im_crop.shape[0] * im_crop.shape[1] * im_crop.max()

#     cv2.imshow("checkbox", im_crop)
#     cv2.waitKey(0)
#     print(max_pix, np.sum(im_crop))
#     print(True if np.sum(im_crop) / max_pix > pix_threshold else False)
# # print(len(rects))
# # print(len(grouping_rects))
# # cv2.imshow(output_image)
