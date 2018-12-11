import csv
from coinmetrics.coinmetrics import CoinMetricsAPI
import utils
import pandas as pd
import time
from datetime import datetime

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
def obtain_data_from_CoinMetrics():

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
			#utils.figure_price(price,asset,l,window)

			#prev_len = len(vector_dict)

			#final_vectors,vector_dict =  next_prev_dict_generation(feature1,feature2,feature3,feature4,price,vector_dict,final_vectors,labels,asset)
			final_vectors,vector_dict =  prev_window_dict_generation(window,feature1,feature2,feature3,feature4,price,vector_dict,final_vectors,labels,asset)
			
			#num_per_asset.append([asset,len(vector_dict) - prev_len])
		
	print("final vectors: " + str(final_vectors))

	return vector_dict

def full_vector(row):
	for item in row:
		if item=='None':
			return 0
	return 1

# convert a csv file to dictionary
# csv must have strictly [asset feature1 feature2 feature3 ..... featureN timestamp or date]  
def convert_csv_to_dict(csv_dir,date_type):

	data = pd.read_csv(csv_dir)

	dict = {}

	columns_num = len(list(data.columns.values))-1

	for row in data.iterrows():
		dt = row[1][columns_num]
		if(date_type == 'timestamp'):

			#print(time.gmtime())
			#dt = time.strftime("%Y-%m-%d", dt)

			a = datetime.fromtimestamp(dt)
			aa = a.strftime("%Y-%m-%d")

			dt = aa

		if(full_vector(row[1])):
			l = []
			for i in range(1,columns_num):
				l.append(str(row[1][i]))
			dict[str(row[1][0]).lower(),str(dt)] = l

	list_of_col_names = list(data.columns.values)
	list_of_col_names.pop()
	list_of_col_names.pop(0)
	
	return dict,list_of_col_names

## merge two dictionaries
def merge_dicts(dict1,dict2):

	merged_dict = {}

	counter = 0
	for key in sorted(dict1.keys()):

		val = dict2.get(key)
		if(val!=None):

			l = dict2[key] + dict1[key]
			ll = []
			for it in l:
				ll.append(str(it))
			merged_dict[key] = ll
			


	return merged_dict

def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

def convert_data_to_timeseries(values,window):

	start = 0
	end = window
	timeseries = []
	timeseries_dictionary = {}
	while(end<len(values)):
		v = values[start:end]
		timeserie = []
		for i in range(0,len(v)):
			if (i!=(len(v) - 1)):
				timeserie.append(v[i][2:-1])
			else:
				timeserie.append(v[i][2:])
				flattened_list = [y for x in timeserie for y in x]
				t = [v[i][0],v[i][1]]
				for x in flattened_list:
					t.append(x)
				timeseries_dictionary[str(t[0]),str(t[1])] = t[2:]
		start += 1
		end += 1

	return timeseries_dictionary


def save_data_as_timeseries(merged_dict,assets_number,window):

	l = []
	for key in sorted(merged_dict.keys()):
		a_list = []
		a_list.append(key[0])
		a_list.append(key[1])
		for x in merged_dict[key]:
			a_list.append(x)
		l.append(a_list)

	timeseries_dictionary = {}

	start = 0
	end = 0
	for i in range(len(assets_number)):
		end = end + int(assets_number[i])
		# normalization of every asset may be here
		t_dict =  convert_data_to_timeseries(l[start:end],window)
		start = end
		timeseries_dictionary =  merge_two_dicts(timeseries_dictionary,t_dict)


	return timeseries_dictionary
		
	
def main():

	#coinmetrics_dict = vector_dict = obtain_data_from_CoinMetrics()

	## writing vector dictionary into csv file
	## sorted assets and timestamps for every asset
	#w = csv.writer(open("../Datasets/all_vectors_np_label.csv", "w"))
	#w = csv.writer(open("../Datasets/all_vectors_prev_window_label.csv", "w"))
	#w.writerow(['asset','blockcount','medianfee','txcount','activeaddresses','label','timestamp'])
	#for key in sorted(vector_dict.keys()):
	#	w.writerow([key[0],vector_dict[key][1],vector_dict[key][2],vector_dict[key][3],vector_dict[key][4],vector_dict[key][5],key[1]])

	csv_dict1,col1_names = convert_csv_to_dict("../Datasets/all_vectors_prev_window_label.csv",'timestamp')
	csv_dict2,col2_names = convert_csv_to_dict("../Datasets/coins_dev_history.csv",'Date')
	csv_dict3,col3_names = convert_csv_to_dict("../Datasets/crypto_ohlcv.csv",'Date')

	merged_dict1 = merge_dicts(csv_dict1,csv_dict2)

	print(len(merged_dict1))

	merged_dict2 = merge_dicts(merged_dict1,csv_dict3)

	col_names_merge = col3_names + col2_names + col1_names

	col_names_new = ['asset','date'] + col_names_merge

	#create col_names for timeseries
	window = 4
	col_names_timeseries = []
	for i in range(0,window):
		for x in col_names_merge[0:-1]:
			col_names_timeseries.append(x + '_' + str(i))

	col_names_timeseries_new =  ['asset','date'] + col_names_timeseries + ['label']


	## create a dictionary of numbers of every asset
	num_dict = {}
	for key in sorted(merged_dict2.keys()):
		val = num_dict.get(key[0])
		if(val!=None):
			num_dict[key[0]] = num_dict[key[0]] + 1;
		else:
			num_dict[key[0]] = 1

	asset_cols = []
	asset_nums = []
	for key in sorted(num_dict.keys()):
		asset_cols.append(str(key))
		asset_nums.append(str(num_dict[key]))

	w = csv.writer(open("../Datasets/assets_number.csv", "w"))
	w.writerow(asset_cols)
	w.writerow(asset_nums)

	
	time_dict = save_data_as_timeseries(merged_dict2,asset_nums,window)

	### the final csv must have the form: asset date feature1 feature2 .... featureN label
	### assets and dates are in increasing order

	### the final dictionary is being converted into csv file
	w = csv.writer(open("../Datasets/all_vectors_merged.csv", "w"))
	w.writerow(col_names_new)
	for key in sorted(merged_dict2.keys()):
		l = [str(key[0]),str(key[1])]
		for x in merged_dict2[key]:
			l.append(str(x))

		w.writerow(l)

	### the final timeseries dictionary is being converted into csv file
	w = csv.writer(open("../Datasets/all_vectors_merged_timeseries.csv", "w"))
	w.writerow(col_names_timeseries_new)
	for key in sorted(time_dict.keys()):
		l = [str(key[0]),str(key[1])]
		for x in time_dict[key]:
			l.append(str(x))

		w.writerow(l)


if __name__ == "__main__":
    main()