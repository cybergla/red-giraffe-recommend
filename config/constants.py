#File paths
FILE_DATA = "./data/prod.csv"	#File to get the training data from
FILE_FEATURES = "./data/features.csv"	#File to get the feature list from
FILE_CLUSTER_MODEL = "./models/cluster_model.pkl"	#Name of the cluster model pickle file
FILE_SCALE_MODEL = "./models/scale_model.pkl"	#Name of the scale model pickle file
FILE_INDEX = "./data/index.csv"		#Name of the index file
FILE_PARTIAL_DATA = "./data/partialdata.csv"#File to perform partial fitting
FILE_SERVER_LOG = "logs/server.log"		#Servel log file
FILE_CLUSTER_LOG = "logs/cluster.log" #Cluster log file
FILE_INDEX_DB = "db/index.db" 		#Index DB location
FILE_INPUT_DATA = "./data/input_data2.csv" 	#Sample input data for debugging purposes

#Index DB stuff
TABLE_NAME = "ids"		#Index table name
COL_INDEX = "indx"		#Column name of index store
COL_ID = "_id"			#Column name of ids

#Preprocessing
SOURCE_TYPE = "csv"		#Type of input data source
SEPARATOR = ","			#Type of separator
SCALER = 'standard'		#Change this to change the type of scaler

#Scaling types
#See this page for more info: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
SCALER_TYPE_STANDARD = 'standard'	#Standard Scaling
SCALER_TYPE_MIN_MAX = 'min-max'		#Min-Max Scaling
SCALER_TYPE_ROBUST = 'robust'		#Robust Scaling

#Clustering constants
#See this page for more info: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans
N_CLUSTERS=-1			#Set to -1 if you want number of clusters to be calculated via CLUSTER_FACTOR
CLUSTER_FACTOR=5		#n_clusters = (size_of_dataset) / (cluster_factor)
INIT='k-means++'		#Init algorithm
BATCH_SIZE=100 			#Size of batches to be processed by Mini Batch K Means
N_INIT=10 				
MAX_NO_OF_IMPROVEMENT=10
VERBOSE=0
RANDOM_STATE=0

#Logging
LOGGER_TOP = 'recommend'				#Top Level logger name
LOGGER_CLUSTER = 'recommend.cluster'	#Cluster logger
LOGGER_PREPROCESS = 'recommend.preprocess'	#Preprocess logger
LOGGER_PARTIAL_FIT = 'recommend.partial_fit'	#Partial Fit logger



