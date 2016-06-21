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

parser = argparse.ArgumentParser(description='Perform K Means clustering on a given dataset.')
parser.add_argument('--input-file', '-i', type=str, help='the file name of the input dataset', default=constants.FILE_DATA)
parser.add_argument('--n-clusters', '-N', type=int, help='number of clusters (default: no. of samples/cf)')
parser.add_argument('--cluster-factor', '-cf', type=int, help='determines number of clusters (default: 5)',default=5)
parser.add_argument('--log', type=str, choices=['DEBUG', 'INFO', 'WARNING','ERROR','CRITICAL'], help='Logging level (default: WARNING)',default="WARNING")
args = parser.parse_args()

log = logging.getLogger('recommend')
log.setLevel(getattr(logging,args.log.upper()))
fh = logging.FileHandler('logs/cluster.log')
fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
log.addHandler(fh)


log.info("Getting data from file")
df = preprocess.get_data(args.input_file,1)

#Normalize data
log.info("Normalizing data")
std_scale = preprocessing.StandardScaler().fit(df)
dataset = std_scale.transform(df)

#Determine number of clusters
if(args.n_clusters == None):
	n_clusters = dataset.shape[0]/args.cluster_factor
else:
	n_clusters = args.n_clusters

#Make model
log.info("Clustering on %s with %d clusters" % (args.input_file,n_clusters))
model = cluster.MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=100, n_init=10, max_no_improvement=10, verbose=0, random_state=0)
model.fit(dataset)
print 'Clustering Sucessfull'
#Save model to disk
log.info("Saving models to disk")
joblib.dump(model, constants.FILE_CLUSTER_MODEL)
#Save normalized model to disk
joblib.dump(std_scale,constants.FILE_SCALE_MODEL)
