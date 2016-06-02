import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def show_PCA_graph(dataset,type="3d"):
	"""
	Does PCA on the dataset and plots pca1 vs pca2
	"""
	pca = PCA(n_components=2)
	transform = pca.fit_transform(dataset)
	#print pca.explained_variance_ratio_
	if type=="3d":	
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(transform[:,0],transform[:,1],range(transform.shape[0]))
	else:
		plt.scatter(transform[:,0],transform[:,1])

	plt.show()
	return

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

DATA_FILE_NAME = "data.csv"

columns = ['index'	,'type'	,'pid'	,'score'	,'id'	,'name'	,'date_entered'	,'date_modified'	,'description'	,'assigned_user_name'	,'deleted'	,'additional_rooms'	,'amenities_status'	,'available_from_date'	,'backend_status'	,'covered_area'	,'currency_type'	,'documents_status'	,'lease_duration_expected'	,'locality'	,'lat'	,'lon'	,'lock_in_period'	,'maintenance_charges'	,'no_of_balconies'	,'no_of_bathrooms'	,'owner_verified'	,'plot_area'	,'posted_on'	,'property_type'	,'property_verified'	,'rejection_comments'	,'rent_expected'	,'security_deposit'	,'state'	,'status'	,'suitable_time_to_call'	,'type_of_accomodation'	,'unit_of_measure'	,'year_of_construction1'	,'city'	,'content_team_status'	,'availabllity'	,'total_floors'	,'property_facing'	,'property_title'	,'unit_of_measure1'	,'floor'	,'lease_type_expected'	,'lease_subtype_expected1'	,'remarks'	,'servant_room'	,'servant_room_with_toilet'	,'flooring_type'	,'pin_code'	,'suitable_day'	,'periodicity_maintenance_bill'	,'proof_of_ownership_of_rental_p']
selected_columns = ['pid', 'name', 'covered_area', 'lease_duration_expected', 'lat', 'lon', 'lock_in_period', 'maintenance_charges', 'no_of_balconies', 'no_of_bathrooms', 'plot_area', 'rent_expected', 'security_deposit', 'type_of_accomodation', 'total_floors', 'floor', 'servant_room', 'servant_room_with_toilet']
df = pd.read_csv(DATA_FILE_NAME,sep=',',header=0,names=selected_columns)
n_samples = df.pid.shape[0]
n_features = len(selected_columns)

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

labels = k_means.labels_
#print len(labels),dataset.shape,train.shape,test.shape
clusters = k_means.predict(test)
for j in xrange(len(test)):
	similar = []
	for i in xrange(len(labels)):
		if labels[i] == clusters[j]:
			similar.append(i+10)
	outf = df.ix[similar]
	outf.to_csv('output'+str(j)+'.csv',columns=selected_columns)

