import glob
import json
import os

import numpy as np
import yaml
from sklearn import cluster


class PipelinesConfig:
    def __init__(self, yaml_path=None):
        """  # NOQA E501
        Helper class to keep all the important config variables in a single place.
        Use `save_` / `load_` functions to store configs in files and load when necessary.
        This class also contains a set `autoconfigure*` functions to automatically build a config based on a ground truth annotations or collection of box sizes.

        Args:
            yaml_path (str, optional): 
                If provided it will try to first load default values for the config and
                then load a saved configuration from a `yaml` file. Defaults to None.
        """
        # Important to adjust these values to match the size of boxes on your image  # NOQA E501
        self.width_range = [(40, 50)]
        self.height_range = [(50, 60)]

        # w/h ratio range for boxes/rectangles filtering
        self.wh_ratio_range = [(0.65, 0.1)]

        # The more scaling factors the more accurate the results
        # but also it takes more time to processing.
        # Too small scaling factor may cause false positives
        # Too big scaling factor will take a lot of processing time
        self.scaling_factors = [1.0]

        # Drawing rectangles
        self.thickness = 2

        # Image processing
        self.dilation_kernel = [(2, 2)]
        # Num of iterations when running dilation tranformation (to engance the image)  # NOQA E501
        self.dilation_iterations = [0]

        self.morph_kernels_type = ['lines']  # 'rectangles'
        self.morph_kernels_thickness = [1]

        # Rectangles grouping
        self.group_size_range = (1, 100)  # minimum number of rectangles in a group, >2 - will ignore groups with single rect  # NOQA E501
        self.vertical_max_distance = [10]  # in pixels
        self.horizontal_max_distance = [self.width_range[0][0] * 2]

        if yaml_path:
            self.load_yaml(yaml_path)

        self.update_num_iterations()

    def update_num_iterations(self):
        """
        This function updates `self.num_iterations` value to match
        the number of configuration sets to be run in the `pipelines`.
        """
        self.num_iterations = 1
        for variable in [
            self.width_range, self.height_range, self.wh_ratio_range,
            self.dilation_kernel,
            self.dilation_iterations, self.morph_kernels_type,
            self.horizontal_max_distance,
            self.morph_kernels_thickness, self.vertical_max_distance
        ]:
            if type(variable) is not list:
                variable = [variable]
            self.num_iterations = len(variable) if self.num_iterations < len(variable) else self.num_iterations  # NOQA E501

    def __conv_to_list(self, x, list_length):
        """Internal/private function.
        Given an object `x` it will return the same object wrapped
        in a list type if `x` is not a list already.

        Args:
            x (object):
                `x` can be any object or a list.
            list_length (int):
                Length of the list to be returned.

        Returns:
            list:
                Returns a list of `x` objects of length `list_length`
        """
        if type(x) is list:
            if len(x) >= list_length:
                return x
            x = x[0]
        return [x for i in range(list_length)]

    def variables_as_iterators(self):
        """Takes a set of variables and converts them into iterators
        based on `self.num_iterations` param.

        Returns:
            zip:
                Configs variables as iterators.
        """
        self.update_num_iterations()

        variables_list = [
            self.width_range, self.height_range, self.wh_ratio_range,
            self.dilation_iterations,
            self.dilation_kernel, self.vertical_max_distance,
            self.horizontal_max_distance,
            self.morph_kernels_type,
            self.morph_kernels_thickness
        ]
        return zip(
            *[self.__conv_to_list(variable, self.num_iterations)
                for variable in variables_list])

    def __calc_margin(
            self, size, margin_percent=0.1, margin_px_limit=5):
        """Calculates by how much given `size` should be extended
        based on provided params.

        Args:
            size (int):
                Height or width value.
            margin_percent (float, optional):
                Float representing margin value in percents.
                Defaults to 0.1.
            margin_px_limit (int, optional):
                Max limit on margin in pixels. Defaults to 5.

        Returns:
            int:
                Margin in pixels.
        """
        calc_margin = int((size * margin_percent))
        return calc_margin if calc_margin < margin_px_limit else margin_px_limit  # NOQA E501

    def autoconfigure(
            self, box_sizes, epsilon=5,
            margin_percent=0.1, margin_px_limit=30,
            use_rect_kernel_for_small=True, rect_kernel_threshold=20):
        """
        Sets config params based on a list of box sizes (h, w).

        Args:
            box_sizes (list):
                List of box sizes. Format: `[(h, w), (h, w)]`
            epsilon (int, optional):
                Epsilon value used for clustering algorithm (DBSCAN).
                Defaults to 5.
            margin_percent (float, optional):
                Float representing margin value in percents.
                Defaults to 0.1.
            margin_px_limit (int, optional):
                Max limit on margin in pixels. Defaults to 30.
            use_rect_kernel_for_small (bool, optional):
                Use `rectangles` kernel for rectangles
                smaller than `rect_kernel_threshold`.
                Defaults to True.
            rect_kernel_threshold (int, optional):
                Threshold for using `rectangles` kernel instead of `lines`.
                Defaults to 20.
        """
        dbscan = cluster.DBSCAN(eps=epsilon, min_samples=1)
        clusters = dbscan.fit_predict(box_sizes)
        box_sizes = np.asarray(box_sizes)

        hw_grouped = []

        for i in range(0, max(clusters) + 1):
            group = box_sizes[clusters == i]

            minh, maxh = (min(group[:, 0]), max(group[:, 0]))
            minw, maxw = (min(group[:, 1]), max(group[:, 1]))

            calc_minh = minh - self.__calc_margin(
                minh, margin_percent, margin_px_limit)
            calc_maxh = maxh + self.__calc_margin(
                maxh, margin_percent, margin_px_limit)
            calc_minw = minw - self.__calc_margin(
                minw, margin_percent, margin_px_limit)
            calc_maxw = maxw + self.__calc_margin(
                maxw, margin_percent, margin_px_limit)

            hw_grouped.append([
                (calc_minw, calc_maxw), (calc_minh, calc_maxh),
                sorted((calc_minw / calc_maxh, calc_maxw / calc_minh)),
                'rectangles' if (
                    use_rect_kernel_for_small
                    and maxh <= rect_kernel_threshold
                    and maxw <= rect_kernel_threshold)
                else 'lines'
            ])
        hw_grouped = np.asarray(hw_grouped)

        self.width_range = hw_grouped[:, 0].tolist()
        self.height_range = hw_grouped[:, 1].tolist()
        self.wh_ratio_range = hw_grouped[:, 2].tolist()
        self.morph_kernels_type = hw_grouped[:, 3].tolist()

        self.update_num_iterations()

    def autoconfigure_from_vott(self, vott_dir, class_tags, **kwargs):
        """
        Reads annotation files from .json format for VoTT and
        automatically creates best config.

        Args:
            vott_dir (string):
                Directory with VoTT annotation files.
            class_tags (list of strings):
                List of tags to search for in annotations.
            **kwargs:
                Check `self.autoconfigure` for available kwargs.
        """
        jsons = glob.glob(os.path.join(vott_dir, "*.json"))

        hw_list = []

        for js in jsons[:]:
            with open(js, mode='r') as js_file:
                for region in json.load(js_file)['regions']:
                    if any(i in region['tags'] for i in class_tags):
                        bbox = region['boundingBox']
                        hw_list.append(
                            (int(bbox['height']), int(bbox['width'])))

        self.autoconfigure(box_sizes=hw_list, **kwargs)

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

        self.update_num_iterations()
