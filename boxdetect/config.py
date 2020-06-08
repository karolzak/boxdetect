# Important to adjust these values to match the size of boxes on your image
width_range = (40, 50)
height_range = (50, 60)

# w/h ratio range for boxes/rectangles filtering
wh_ratio_range = (0.65, 0.1)

# Rectangular kernels border thickness
border_thickness = 1

# The more scaling factors the more accurate the results
# but also it takes more time to processing.
# Too small scaling factor may cause false positives
# Too big scaling factor will take a lot of processing time
scaling_factors = [0.5]

# Drawing rectangles
thickness = 2

# Image enhancement
dilation_kernel = (2, 2)
# Num of iterations when running dilation tranformation (to engance the image)
dilation_iterations = 0

# Rectangles grouping
group_size_range = (1, 100)  # minimum number of rectangles in a group, >2 - will ignore groups with single rect  # NOQA E501
vertical_max_distance = 10  # in pixels
# Multiplier to be used with mean width of all the rectangles detected
# E.g. for multiplier of 4 the maximum distance between boxes to be grouped together will be:  # NOQA E501
# max_horizontal_distance = np.mean(all_rect_widths) * 3
horizontal_max_distance_multiplier = 3
