import yaml


class PipelinesConfig:
    def __init__(self, yaml_path=None):
        # Important to adjust these values to match the size of boxes on your image  # NOQA E501
        self.width_range = (40, 50)
        self.height_range = (50, 60)

        # w/h ratio range for boxes/rectangles filtering
        self.wh_ratio_range = (0.65, 0.1)

        # The more scaling factors the more accurate the results
        # but also it takes more time to processing.
        # Too small scaling factor may cause false positives
        # Too big scaling factor will take a lot of processing time
        self.scaling_factors = [0.5]

        # Drawing rectangles
        self.thickness = 2

        # Image processing
        self.dilation_kernel = (2, 2)
        # Num of iterations when running dilation tranformation (to engance the image)  # NOQA E501
        self.dilation_iterations = 0

        self.morph_kernels_type = 'lines'  # 'rectangles'
        self.morph_kernels_lines_length = 15
        self.morph_kernels_lines_thickness = 1
        # Rectangular kernels border thickness
        self.border_thickness = 1

        # Rectangles grouping
        self.group_size_range = (1, 100)  # minimum number of rectangles in a group, >2 - will ignore groups with single rect  # NOQA E501
        self.vertical_max_distance = 10  # in pixels
        # Multiplier to be used with mean width of all the rectangles detected
        # E.g. for multiplier of 4 the maximum distance between boxes to be grouped together will be:  # NOQA E501
        # max_horizontal_distance = np.mean(all_rect_widths) * 3
        self.horizontal_max_distance = self.width_range[0] * 2

        if yaml_path:
            self.load_yaml(yaml_path)

    def save_yaml(self, path):
        variables_dict = {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith('__') and not callable(key)
        }
        with open(r'%s' % path, 'w') as file:
            yaml.dump(variables_dict, file)

    def load_yaml(self, path, suppress_warnings=False):
        with open(r'%s' % path, 'r') as file:
            variables_dict = yaml.load(file, Loader=yaml.FullLoader)

        for key, value in variables_dict.items():
            if not suppress_warnings and not key in self.__dict__.keys():
                print("WARNING: Loaded variable '%s' which was not previously present in the config." % key)  # NOQA E501
            setattr(self, key, value)
