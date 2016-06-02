import numpy as np
import pandas as pd
from sklearn import cluster
from matplotlib import pyplot as plt

names = ['pid', 'name', 'covered_area', 'lease_duration_expected', 'lat', 'lon', 'lock_in_period', 'maintenance_charges', 'no_of_balconies', 'no_of_barooms', 'plot_area', 'rent_expected', 'security_deposit', 'type_of_accomodation', 'total_floors', 'floor', 'servant_room', 'servant_room_wi_toilet']
df = pd.read_csv('data.csv',sep=',',header=1,names=names)
n_samples = df.pid.unique().shape[0]
n_features = len(names)

#print "Samples= ",n_samples
#print "Features= ",n_features

dataset = np.zeros((n_samples,n_features-3))
for row in df.itertuples():
	for x in xrange(0,n_features-3):
		dataset[row[0],x] = row[x+3]

train = dataset.copy()
train = train[10:]
test = dataset.copy()
test = test[:10]

k_means = cluster.KMeans(n_clusters=10)
k_means.fit(train)

def elbow_test(c_start,c_end,step):
	"""
	Plots graph of inertia vs no of clusters
	"""
	inertia = []
	for i in xrange(c_start,c_end,step):
		k_means = cluster.KMeans(n_clusters=i)
		k_means.fit(train)
		inertia.append(k_means.inertia_)
	plt.plot(range(c_start,c_end,step),inertia)
	plt.xlabel("Number of Clusters")
	plt.ylabel("Inertia")
	plt.show()
	return

elbow_test(1,50,1)


"""
labels = k_means.labels_
#print len(labels),dataset.shape,train.shape,test.shape
clusters = k_means.predict(test)
for j in xrange(len(test)):
	similar = []
	for i in xrange(len(labels)):
		if labels[i] == clusters[j]:
			similar.append(i+10)
	outf = df.ix[similar]
	outf.to_csv('output'+str(j)+'.csv',columns=names)
"""