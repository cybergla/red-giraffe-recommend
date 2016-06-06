import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn.externals import joblib
from sklearn import preprocessing

FILE_DATA = "data.csv"
FILE_FEATURES = "features.csv"

df_features = pd.read_csv(FILE_FEATURES,sep=',',header=None)
selected_columns = df_features[0].values

df = pd.read_csv(FILE_DATA,sep=',',header=0, usecols=selected_columns)

#TODO: preprocessing input, validation

std_scale = preprocessing.StandardScaler().fit(df[selected_columns[2:]])
joblib.dump(std_scale,'scale_model.pkl')
dataset = std_scale.transform(df[selected_columns[2:]])

n_clusters = dataset.shape[0]/5
model = cluster.MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=100, n_init=10, max_no_improvement=10, verbose=0, random_state=0)
model.fit(dataset)

#Save model to disk
joblib.dump(model, 'cluster_model.pkl')
