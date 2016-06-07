import numpy as np
import pandas as pd
import re
from sklearn import cluster
from sklearn.externals import joblib
from sklearn import preprocessing

import constants
import preprocess

df = preprocess.process_data('out.csv')
selected_columns = preprocess.get_features()

#Normalize data
std_scale = preprocessing.StandardScaler().fit(df[selected_columns[2:]])
dataset = std_scale.transform(df[selected_columns[2:]])


#Determine number of clusters
n_clusters = dataset.shape[0]/5
#Make model
model = cluster.MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=100, n_init=10, max_no_improvement=10, verbose=0, random_state=0)
model.fit(dataset)

#Save model to disk
joblib.dump(model, 'cluster_model.pkl')
#Save normalized model to disk
joblib.dump(std_scale,'scale_model.pkl')