import numpy as np
import pandas as pd
import re
from sklearn import cluster
from sklearn.externals import joblib
from sklearn import preprocessing

FILE_DATA = "result.csv"	#File to get the training data from
FILE_FEATURES = "features.csv"	#File to get the feature list from

df_features = pd.read_csv(FILE_FEATURES,sep=',',header=None)	
selected_columns = df_features[0].values	#Only select these features

df = pd.read_csv(FILE_DATA,sep=',',header=0, usecols=selected_columns)
#Remove rows with NaN, null values
df = df.dropna()	

#Preprocess data
for index, row in df.iterrows():
	#Replace bhk, rk etc. in type of accomodation
	df.loc[index,"_source/type_of_accomodation"] = re.sub(r"\D*","",df.loc[index,"_source/type_of_accomodation"])
	#Replace floor desc with number
	df.loc[index,"_source/floor"] = re.sub(r"Duplex|Ground_\d|Ground",'0',df.loc[index,"_source/floor"])
	df.loc[index,"_source/floor"] = re.sub(r"\D*","",df.loc[index,"_source/floor"])

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

