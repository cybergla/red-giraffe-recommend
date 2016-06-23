import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn.externals import joblib
from sklearn import preprocessing
import logging

import utils.preprocess as preprocess
import config.constants as constants

def fit(input_file=constants.FILE_PARTIAL_DATA):
	log = logging.getLogger('recommend.partial_fit')

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


	

	
