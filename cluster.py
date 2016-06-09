import numpy as np
import pandas as pd
import sys
from sklearn import cluster
from sklearn.externals import joblib
from sklearn import preprocessing

import constants
import preprocess

if (len(sys.argv) > 1):
	input_file = sys.argv[1]
else:
	input_file = constants.FILE_DATA

df = preprocess.get_data(input_file,1)

#Normalize data
std_scale = preprocessing.StandardScaler().fit(df)
dataset = std_scale.transform(df)

#Determine number of clusters
n_clusters = dataset.shape[0]/5

#Make model
model = cluster.MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=100, n_init=10, max_no_improvement=10, verbose=0, random_state=0)
model.fit(dataset)

#Save model to disk
joblib.dump(model, constants.FILE_CLUSTER_MODEL)
#Save normalized model to disk
joblib.dump(std_scale,constants.FILE_SCALE_MODEL)