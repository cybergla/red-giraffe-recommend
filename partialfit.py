import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn.externals import joblib
from sklearn import preprocessing
import logging

import utils.preprocess as preprocess
import config.constants as constants

def main(input_file=constants.FILE_PARTIAL_DATA):
	log.info("Partial fit on %s" % input_file)
	df = preprocess.get_data(input_file,1,to_append=True)
	log.debug("Size of dataset: %d x %d" % df.shape)

	#Load scale model from disk
	std_scale = joblib.load(constants.FILE_SCALE_MODEL)
	dataset = std_scale.transform(df)

	#Load cluster model from disk
	model = joblib.load(constants.FILE_CLUSTER_MODEL)
	lists=model.labels_
	log.debug("Old labels size: %d" % lists.shape)

	#Partial fit on the old trained model
	model.partial_fit(dataset)

	#Append the labels
	lists=np.append(lists,model.labels_)
	model.labels_=lists

	log.debug("New labels size: %d" % lists.shape)
	#Save updated cluster model to disk
	joblib.dump(model, constants.FILE_CLUSTER_MODEL)

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Partially fit a small dataset to an existing model')
	parser.add_argument('--input-file', '-i', type=str, help='the file name of the input dataset', default=constants.FILE_PARTIAL_DATA)
	parser.add_argument('--log', type=str, choices=['DEBUG', 'INFO', 'WARNING','ERROR','CRITICAL'], help='Logging level (default: WARNING)',default="WARNING")

	args = parser.parse_args()

	log = logging.getLogger('recommend')
	log.setLevel(getattr(logging,args.log.upper()))
	fh = logging.FileHandler('logs/partial_fit.log')
	fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
	log.addHandler(fh)

	main(input_file=args.input_file)

	

	
