import yaml


class PipelinesConfig:
    def __init__(self, yaml_path=None):
        """  # NOQA E501
        Helper class to keep all the important config variables in a single place.
        Use `save_` / `load_` functions to store configs in files and load when necessary.

        Args:
            yaml_path (str, optional): 
                If provided it will try to first load default values for the config and
                then load a saved configuration from a `yaml` file. Defaults to None.
        """
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
        self.horizontal_max_distance = self.width_range[0] * 2

        if yaml_path:
            self.load_yaml(yaml_path)

    def save_yaml(self, path):
        """
        Saves current config into `yaml` file based on provided `path`.

        Args:
            path (str):
                Path to the file to save config.
        """
        variables_dict = {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith('__') and not callable(key)
        }
        with open(r'%s' % path, 'w') as file:
            yaml.dump(variables_dict, file)

    def load_yaml(self, path, suppress_warnings=False):
        """
        Loads configuration from `yaml` file based on provided `path`.

        Args:
            path (str):
                Path to the file to load config from.
            suppress_warnings (bool, optional):
                To show or not show warnings about potential mismatches.
                Defaults to False.
        """
        with open(r'%s' % path, 'r') as file:
            variables_dict = yaml.load(file, Loader=yaml.FullLoader)

        for key, value in variables_dict.items():
            if not suppress_warnings and key not in self.__dict__.keys():
                print("WARNING: Loaded variable '%s' which was not previously present in the config." % key)  # NOQA E501
            setattr(self, key, value)
