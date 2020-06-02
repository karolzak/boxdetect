import argparse
import glob
import sys
sys.path.append("../.")
from functools import partial
from multiprocess import Pool
# from multiprocessing import Pool

import cv2
import imutils
import numpy as np

from box_detector import config
from box_detector.pipelines import process_image


if __name__ == "__main__":		
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--image_dir", required=True,
		help="path to the input dir")
	ap.add_argument("-p", "--plot", type=bool, default=False, required=False,
		help="plot results")
	ap.add_argument("-ps", "--pool_size", type=int, default=1, required=False,
		help="pool size for multiprocessing")
	args = vars(ap.parse_args())

	images = glob.glob(args["image_dir"] + '*')
	plot = bool(args["plot"])
	pool_size = int(args["pool_size"])
	print(args["image_dir"], plot)

	pool = Pool(pool_size)
	pool_outputs = pool.map(
        partial(process_image, config=config, plot=plot),
        images[:]
    )
	pool.close()
	pool.join()
	pool.terminate()
	for output in pool_outputs:		
		cv2.imwrite(output[2].replace('\\in','\\out'), output[3])
