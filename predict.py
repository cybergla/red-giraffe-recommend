import numpy as np
import pandas as pd
import re
from sklearn import cluster
from sklearn.externals import joblib
from sklearn import preprocessing
import preprocess

#Preprocess data
df = preprocess.process_data('input_data.csv')

#Get list of features
selected_columns = preprocess.get_features()

#Load scale model from disk
std_scale = joblib.load('scale_model.pkl')
#Normalize data
test = std_scale.transform(df[selected_columns[2:]])
#Load cluster model from disk
model = joblib.load('cluster_model.pkl')
#Get labels of cluster model
labels = model.labels_
#Predict cluster no. of test data
clusters = model.predict(test)
df_in = pd.read_csv("result.csv",sep=',',header=0)

for j in xrange(len(test)):
		similar = []
		for i in xrange(len(labels)):
		       if labels[i] == clusters[j]:
		               similar.append(i)
		outf = df_in.ix[similar]
		outf.to_csv('./results/output'+str(j)+'.csv')