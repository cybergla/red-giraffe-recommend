import numpy as np
import pandas as pd
import argparse
from sklearn import cluster
from sklearn.externals import joblib
from sklearn import preprocessing
import logging

#Suppress warnings from clustering
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import config.constants as constants
import utils.preprocess as preprocess
import utils.scaling as scaling
def fit(input_file=constants.FILE_DATA):
	log = logging.getLogger(constants.LOGGER_CLUSTER)

	log.info("Getting data from file")
	df = preprocess.get_data(input_file,1)

	#Normalize data
	log.info("Normalizing data")
	#Normalize data
	dataset = scaling.scale_data(df)
	#Determine number of clusters
	if(constants.N_CLUSTERS == -1):
		n_clusters = dataset.shape[0]/constants.CLUSTER_FACTOR
	else:
		n_clusters = constants.N_CLUSTERS

	#Make model
	log.info("Clustering on %s with %d clusters" % (input_file,n_clusters))
	model = cluster.MiniBatchKMeans(init=constants.INIT, n_clusters=n_clusters, batch_size=constants.BATCH_SIZE, n_init=constants.N_INIT, max_no_improvement=constants.MAX_NO_OF_IMPROVEMENT, verbose=constants.VERBOSE, random_state=constants.RANDOM_STATE)
	model.fit(dataset)

	print 'Clustering successful'

	#Save model to disk
	log.info("Saving models to disk")
	joblib.dump(model, constants.FILE_CLUSTER_MODEL)

	return
