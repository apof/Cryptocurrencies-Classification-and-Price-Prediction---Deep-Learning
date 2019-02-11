import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import csv
import utils
from sklearn import preprocessing
import math
from scipy import stats


DATA_DIR = "../Datasets/all_vectors_merged.csv"
ASSETS_DIR = "../Datasets/assets_number.csv"
DATA_DIR_2 = "../Datasets/Data_to_merge/crypto_ohlcv.csv"

def plot_features(data):

	assets_number = utils.load_dataset(ASSETS_DIR)

	number_per_asset = []
	for row in assets_number.iterrows():
		for i in range(0,len(row[1])):
			number_per_asset.append(row[1][i])

	asset_frames = []
	start = 0

	for i in range(0,len(number_per_asset)):
		end = start + number_per_asset[i]
		frame = pd.DataFrame()
		frame = data[start:end]
		asset_frames.append(frame)
		start = end

	Fet_names = []
	Fet_names.append('Open')
	Fet_names.append('High')
	Fet_names.append('Low')
	Fet_names.append('Close')
	Fet_names.append('Volume')
	Fet_names.append('MarketCup')
	Fet_names.append('Forks')
	Fet_names.append('Stars')
	Fet_names.append('Subscribers')
	Fet_names.append('Total_issues')
	Fet_names.append('Closed_issues')
	Fet_names.append('Pull_requests_merged')
	Fet_names.append('Pull_request_contributors')
	Fet_names.append('Commit_count_4_weeks')
	Fet_names.append('Blockcount')
	Fet_names.append('Mediafee')
	Fet_names.append('Txcount')
	Fet_names.append('Activeaddresses')

	Fets = []
	for i in range(0,len(Fet_names)):
		Fets.append([])

	Indexes = []
	index = 1

	for row in asset_frames[1].iterrows():
		for i in range(0,len(Fets)):
			Fets[i].append(row[1][i+2])
		Indexes.append(index)
		index += 1

	i=3
	plt.plot(Indexes,Fets[i])
	plt.xlabel('data index')
	plt.ylabel(Fet_names[i])
	plt.savefig('Plots_Results/'+Fet_names[i]+'_btc.png')
