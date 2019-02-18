import csv
from coinmetrics.coinmetrics import CoinMetricsAPI
import utils
import pandas as pd
import time
from datetime import datetime
import math
from sklearn import preprocessing
import math
from scipy import stats


def full_vector(row):
	for item in row:
		if(item=='None' or item=='-'):
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

def data_normalize(data):

	dim = len(data[0])
	num = len(data)

	normalized_list = []
	for i in range(0,num):
		normalized_list.append([])

	mean_val = []
	for i in range(0,dim):
		sum = 0.0
		flag = 1
		for d in data:
				if(i!=0 and i!=1 and i!=(dim-1)):
					sum += float(d[i])
				else:
					flag = 0
		if(flag!=0):
			mean_val.append(sum/num)


	st = []
	for i in range(0,dim):
		sum = 0.0
		flag = 1
		for d in data:
			if(i!=0 and i!=1 and i!=(dim-1)):
				sum += (float(d[i]) - mean_val[i-2])*(float(d[i]) - mean_val[i-2])
			else:
				flag = 0

		if(flag!=0):
			st.append(math.sqrt(sum/(num-1)))


	for i in range(0,num):
		for j in range(0,dim):
			if(j==0 or j==1 or j==(dim-1)):
				normalized_list[i].append(data[i][j])
			else:
				if(st[j-2]==0):
					normalized_list[i].append(st[j-2])
				else:
					normalized_list[i].append((float(data[i][j])-mean_val[j-2])/st[j-2])

	return normalized_list



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
		normalized_data = data_normalize(l[start:end])
		t_dict =  convert_data_to_timeseries(normalized_data,window)
		start = end
		timeseries_dictionary =  merge_two_dicts(timeseries_dictionary,t_dict)


	return timeseries_dictionary


def get_bitcoin_batch(data):

	data = data[data.Volume != '-']
	data = data[data.MarketCap != '-']

	data.set_index("Symbol", inplace=True)
	btc_frame = data.loc['BTC']

	columnsTitles=["Open","High","Close","Volume","MarketCap","Low"]
	btc_frame=btc_frame.reindex(columns=columnsTitles)
	print btc_frame["Low"].mean()
	print btc_frame["Low"].std()

	return btc_frame


def data_scaling(df,scaling_method):
	
	#returns a numpy array
	x = df.values

	#check that standarization is correct
	#transposed_df = df.transpose()
	#xx = transposed_df.values
	#print stats.zscore(xx[0])

	if(scaling_method == 'standarization'):
		res = pd.DataFrame(preprocessing.StandardScaler().fit_transform(x),columns=df.columns, index=df.index)
	elif(scaling_method == 'normalization'):
		res = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(x),columns=df.columns, index=df.index)

	return res

def write_data(data_frame):

	values = data_frame.values
	
	w = csv.writer(open("../Datasets/Final_Data/regression_data_btc.csv", "w"))

	cols = ['Open','High','Close','Volume','MarketCup','Low']
	w.writerow(cols)

	for v in values:
		l=[]
		for dat in v:
			l.append(str(dat))
		w.writerow(l)

def main():

	flag = 1

	if(flag==0):

		csv_dict1,col1_names = convert_csv_to_dict("../Datasets/Data_to_merge/all_vectors_prev_window_label.csv",'timestamp')
		csv_dict2,col2_names = convert_csv_to_dict("../Datasets/Data_to_merge/coins_dev_history.csv",'Date')
		csv_dict3,col3_names = convert_csv_to_dict("../Datasets/Data_to_merge/crypto_ohlcv.csv",'Date')

		merged_dict1 = merge_dicts(csv_dict1,csv_dict2)

		merged_dict2 = merge_dicts(merged_dict1,csv_dict3)

		print("Vectors obdained finally: " + str(len(merged_dict2)))

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

		### the final dictionary is being converted into csv file
		w = csv.writer(open("../Datasets/all_vectors_merged.csv", "w"))
		w.writerow(col_names_new)
		for key in sorted(merged_dict2.keys()):
			l = [str(key[0]),str(key[1])]
			for x in merged_dict2[key]:
				l.append(str(x))

			w.writerow(l)

		time_dict = save_data_as_timeseries(merged_dict2,asset_nums,window)

		print("Timeseries Vectors obdained finally: " + str(len(time_dict)))

		### the final csv must have the form: asset date feature1 feature2 .... featureN label
		### assets and dates are in increasing order
		### the final timeseries dictionary is being converted into csv file
		w = csv.writer(open("../Datasets/normalized_all_vectors_merged_timeseries.csv", "w"))
		w.writerow(col_names_timeseries_new)
		for key in sorted(time_dict.keys()):
			l = [str(key[0]),str(key[1])]
			for x in time_dict[key]:
				l.append(str(x))

			w.writerow(l)

	else:

		data = utils.load_dataset("../Datasets/Data_to_merge/crypto_ohlcv.csv")
		btc_data = get_bitcoin_batch(data)
		btc_scaled_data = data_scaling(btc_data,'normalization')

		write_data(btc_scaled_data)


if __name__ == "__main__":
    main()