import csv
from coinmetrics.coinmetrics import CoinMetricsAPI
import utils
import pandas as pd
import time
from datetime import datetime
import math


## labelling taking into account the previous and the next day value
def next_prev_labeling(price):
	labels = []
	l = []
	for i in range(1,(len(price)-1)):
		val = (price[i+1][1] - price[i][1]) - (price[i][1] - price[i-1][1])
		if(val>0):
			label = 1 
		else:
			label = 0
		labels.append([price[i][0],label])
		l.append(label)

	return labels,l

## creating a dictionary for next prev labelling method
def next_prev_dict_generation(feature1,feature2,feature3,feature4,price,vector_dict,final_vectors,labels,asset):
	dict1 = {}
	dict2 = {}
	dict3 = {}
	dict4 = {}

	for x in feature1:
		dict1[x[0]] = x[1]
	for x in feature2:
		dict2[x[0]] = x[1]
	for x in feature3:
		dict3[x[0]] = x[1]
	for x in feature4:
		dict4[x[0]] = x[1]

	## selecting timestamps that have all the features with non zero values
	## and place them into a dictionary

	for i in range(0,(len(price)-2)):

		v1 = dict1.get(price[i+1][0])
		v2 = dict2.get(price[i+1][0])
		v3 = dict3.get(price[i+1][0])
		v4 = dict4.get(price[i+1][0])

		if all([v1 != None,v2 != None,v3 != None,v4 != None]):

			#print(str(asset) + " " + str(price[i+1][0]) + " " + str(price[i+1][1]) + " " + str(v1) + " " + str(v2) + " " + str(v3) + " " + str(v4) + " " + str(labels[i][1]))
			vector_dict[str(asset),str(price[i+1][0])] = [str(price[i+1][1]),str(v1),str(v2),str(v3),str(v4),str(labels[i][1])]
			final_vectors = final_vectors + 1

	#print("---------------------")

	return final_vectors,vector_dict


## compute and return mean price of a given window
def compute_prev_window_median(window):
	sum = 0.0
	for i in range(0,len(window)):
		sum = sum + window[i][1]

	return sum/len(window)


## labelling taking into account the mean price of a given previous day-window
def prev_window_labeling(price,window):
	labels = []
	l = []
	w_start = 0
	w_end = window-1
	for i in range(window,len(price)):
		median_prev = compute_prev_window_median(price[w_start:w_end])
		if(median_prev<price[i][1]):
			label = 1 
		else:
			label = 0
		labels.append([price[i][0],label])
		l.append(label)
		w_start += 1
		w_end += 1

	return labels,l


## creating a dictionary for prev window labelling method 
def prev_window_dict_generation(window,feature1,feature2,feature3,feature4,price,vector_dict,final_vectors,labels,asset):


	dict1 = {}
	dict2 = {}
	dict3 = {}
	dict4 = {}

	for x in feature1:
		dict1[x[0]] = x[1]
	for x in feature2:
		dict2[x[0]] = x[1]
	for x in feature3:
		dict3[x[0]] = x[1]
	for x in feature4:
		dict4[x[0]] = x[1]

	## selecting timestamps that have all the features with non zero values
	## and place them into a dictionary

	for i in range(window,len(price)):

		v1 = dict1.get(price[i][0])
		v2 = dict2.get(price[i][0])
		v3 = dict3.get(price[i][0])
		v4 = dict4.get(price[i][0])

		if all([v1 != None,v2 != None,v3 != None,v4 != None]):

			vector_dict[str(asset),str(price[i][0])] = [str(price[i][1]),str(v1),str(v2),str(v3),str(v4),str(labels[i-window][1])]
			final_vectors = final_vectors + 1

	#print("---------------------")


	return final_vectors,vector_dict


# obdain all data of all assets which have features blockcount/medianfee/txcount/activeaddresses
# choose an appropriate labelling in order to label them
def obdain_data_from_CoinMetrics():

	cm = CoinMetricsAPI()

	final_vectors = 0
	vector_dict = {}

	num_per_asset = []

	supported_assets = cm.get_supported_assets()

	for asset in supported_assets:
		cm.get_available_data_types_for_asset(asset)

		avail_data_types = cm.get_available_data_types_for_asset(asset)

		count = 0

		if 'blockcount' in avail_data_types:
			count = count + 1
		if 'medianfee' in avail_data_types:
			count = count + 1
		if 'txcount' in avail_data_types:
			count = count + 1
		if 'activeaddresses' in avail_data_types:
			count = count + 1

		## check only assets that have available all the features
		if(count == 4):

			print("------------------> " + asset)

			# timestamps start: 2009-01-09 end: 2018-11-11 --> chosen from btc
			# range in timestamp style 1231459201 - 1541980799

			feature1 = cm.get_asset_data_for_time_range(asset, 'blockcount', 1231459201, 1541980799)
			feature2 = cm.get_asset_data_for_time_range(asset, 'medianfee', 1231459201, 1541980799)
			feature3 = cm.get_asset_data_for_time_range(asset, 'txcount', 1231459201, 1541980799)
			feature4 = cm.get_asset_data_for_time_range(asset, 'activeaddresses', 1231459201, 1541980799)

			price = cm.get_asset_data_for_time_range(asset, 'price(usd)', 1231459201, 1541980799)

			window = 30
			#labels,l = next_prev_labeling(price)
			labels,l = prev_window_labeling(price,window)
			if(asset=='btc' or asset=='eth'):
				utils.figure_and_save_price(price,asset,l,window)

			#prev_len = len(vector_dict)

			#final_vectors,vector_dict =  next_prev_dict_generation(feature1,feature2,feature3,feature4,price,vector_dict,final_vectors,labels,asset)
			final_vectors,vector_dict =  prev_window_dict_generation(window,feature1,feature2,feature3,feature4,price,vector_dict,final_vectors,labels,asset)
			
			#num_per_asset.append([asset,len(vector_dict) - prev_len])
		
	print("Vectors obdained from CoinMetricsAPI: " + str(final_vectors))


	return vector_dict

def main():

	vector_dict = obdain_data_from_CoinMetrics()

	## writing vector dictionary into csv file
	## sorted assets and timestamps for every asset
	#w = csv.writer(open("../Datasets/all_vectors_np_label.csv", "w"))
	w = csv.writer(open("../Datasets/Data_to_merge/all_vectors_prev_window_label.csv", "w"))
	w.writerow(['asset','blockcount','medianfee','txcount','activeaddresses','label','timestamp'])
	for key in sorted(vector_dict.keys()):
		w.writerow([key[0],vector_dict[key][1],vector_dict[key][2],vector_dict[key][3],vector_dict[key][4],vector_dict[key][5],key[1]])

if __name__ == "__main__":
    main()