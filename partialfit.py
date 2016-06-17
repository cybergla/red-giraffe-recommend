import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn.externals import joblib
from sklearn import preprocessing

import utils.preprocess as preprocess
import config.constants as constants

def main(input_file=constants.FILE_PARTIAL_DATA):
	df = preprocess.get_data(input_file,1,to_append=True)

	#Load scale model from disk
	std_scale = joblib.load(constants.FILE_SCALE_MODEL)
	dataset = std_scale.transform(df)

	#Load cluster model from disk
	model = joblib.load(constants.FILE_CLUSTER_MODEL)
	lists=model.labels_
	#Partial fit on the old trained model
	model.partial_fit(dataset)
	#Append the labels
	lists=np.append(lists,model.labels_)
	model.labels_=lists
	#Save updated cluster model to disk
	joblib.dump(model, constants.FILE_CLUSTER_MODEL)

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Partially fit a small dataset to an existing model')
	parser.add_argument('--input-file', '-i', type=str, help='the file name of the input dataset', default=constants.FILE_PARTIAL_DATA)

	args = parser.parse_args()
	main(input_file=args.input_file)

	

	
