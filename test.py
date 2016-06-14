#!/usr/local/bin/python
import os
import pandas as pd
import argparse

import predict
import utils.preprocess as preprocess
import config.constants as constants

parser = argparse.ArgumentParser(description='Test the workflow. Perform K Means clustering, then run predictions for N rows of the dataset. Predictions are stored in results/output(x).csv')
parser.add_argument('--train-file', '-i', type=str, help='the file name of the training dataset', default=constants.FILE_DATA)
parser.add_argument('--test_file','-t',type=str,help='the file name of the testing dataset',default=constants.FILE_DATA)
parser.add_argument('-N', type=int, help='number of testing samples to select from the testing dataset (default: 5)',default=5)
args = parser.parse_args()

os.system("python cluster.py -i " + args.train_file)

selected_columns = preprocess.get_features()
df = pd.read_csv(args.test_file,usecols=selected_columns)
for i in xrange(args.N):
	predict.get_reccomended_ids(df[i:i+1],args.test_file,None,False).to_csv("./results/output "+str(i)+".csv",index=False)