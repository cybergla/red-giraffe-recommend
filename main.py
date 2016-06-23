import argparse
import logging

import config.constants as constants
import cluster
import partialfit

parser = argparse.ArgumentParser(description='Perform K Means clustering on a given dataset.')
parser.add_argument('--input-file', '-i', type=str, help='the file name of the input dataset', default=constants.FILE_DATA)
parser.add_argument('--log', type=str, choices=['DEBUG', 'INFO', 'WARNING','ERROR','CRITICAL'], help='Logging level (default: WARNING)',default="WARNING")
parser.add_argument('--mode','-m', type=str, choices=['FULL','PARTIAL'], help='Type of clustering (default: FULL)',default="FULL")
args = parser.parse_args()

log = logging.getLogger('recommend')
log.setLevel(getattr(logging,args.log.upper()))
fh = logging.FileHandler(constants.FILE_CLUSTER_LOG)
fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
log.addHandler(fh)

if args.mode == "FULL":
	cluster.fit(args.input_file)
elif args.mode == "PARTIAL":
	partialfit.fit(args.input_file)