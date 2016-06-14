import numpy as np
import pandas as pd
import argparse
from sklearn import cluster
from sklearn.externals import joblib
from sklearn import preprocessing
#Suppress warnings from clustering
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import config.constants as constants
import utils.preprocess as preprocess

parser = argparse.ArgumentParser(description='Perform K Means clustering on a given dataset.')
parser.add_argument('--input-file', '-i', type=str, help='the file name of the input dataset', default=constants.FILE_DATA)
parser.add_argument('--n-clusters', '-N', type=int, help='number of clusters (default: no. of samples/5)')

args = parser.parse_args()

df = preprocess.get_data(args.input_file,1)

#Normalize data
std_scale = preprocessing.StandardScaler().fit(df)
dataset = std_scale.transform(df)

#Determine number of clusters
if(args.n_clusters == None):
	n_clusters = dataset.shape[0]/5
else:
	n_clusters = args.n_clusters

#Make model
model = cluster.MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=100, n_init=10, max_no_improvement=10, verbose=0, random_state=0)
model.fit(dataset)

#Save model to disk
joblib.dump(model, constants.FILE_CLUSTER_MODEL)
#Save normalized model to disk
joblib.dump(std_scale,constants.FILE_SCALE_MODEL)