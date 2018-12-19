import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import csv
from datetime import datetime
import time



def convert_data_to_arrays(data,flag):

    list_of_vectors = []
    list_of_labels = []


    for row in data.iterrows():

        #row[0] is record id 1,2,3...
        #row[1][0] contains the asset
        #row[1][1] contains the date
        #row[2][0-(last-2)] contains all features
        #row[1][last-1] contains the label

        vector = []

        for i in range(2,(len(row[1])-2)):
            vector.append(row[1][i])

        arr = np.array(vector)
        arr = arr.astype(np.float32)
        list_of_vectors.append(arr)

        if(flag==0):
            if(int(row[1][(len(row[1])-2)])==0):
                label = [0,1]
            else:
                label = [1,0]
        else:
            if(int(row[1][(len(row[1])-2)])==0):
                label = 0
            else:
                label = 1

        arr = np.array(label)
        arr = arr.astype(np.float32)
        list_of_labels.append(arr)

    feature_num = len(row[1])-4

    return np.array(list_of_vectors),np.array(list_of_labels), feature_num

def load_dataset(dir_name):
	
	data = pd.read_csv(dir_name, sep=",")
	suffled_data = data.sample(frac=1)
	return data


def smash_train_test(df):

	df['split'] = np.random.randn(df.shape[0], 1)
	msk = np.random.rand(len(df)) <= 0.8

	train = df[msk]
	test = df[~msk]

	return train,test


def next_batch(data, num_of_batch,batch_size):
  start_index = num_of_batch*batch_size
  end_index = num_of_batch*batch_size + batch_size
  if(end_index > len(data)):
    end_index = len(data)
  return data[start_index:end_index]


def figure_and_save_price(price,asset,label,window):
    x = []
    y = []
    c = np.array(['red','green'])
    
    if(window==0):
        for i in range(1,(len(price)-1)):
            x.append(price[i][0])
            y.append(price[i][1])
    else:
        for i in range(window,(len(price))):
            x.append(price[i][0])
            y.append(price[i][1])
    
    w = csv.writer(open("../Datasets/btc_prices.csv", "w"))
    w.writerow(['date','price'])
    for i in range(len(x)):
        dat = [str(x[i]),str(y[i])]
        w.writerow(dat)    
            
    fig = plt.figure(figsize=(8,8))
    plt.scatter(x, y, c=label, cmap=matplotlib.colors.ListedColormap(c))
    fig.savefig('Plots/' + asset + '_plot_labels.png')

    cb = plt.colorbar()
    loc = np.arange(0,max(label),max(label)/float(len(c)))
    cb.set_ticks(loc)
    cb.set_ticklabels(c)


def examine_faults(preds,next_label):

    Accuracy = 0

    l = []

    for i in range(len(preds[0])):
        lb1 = 0
        if(preds[0][i][0] > preds[0][i][1]):
            lb1 = 1
        lb2 = 0
        if(next_label[i][0] > next_label[i][1]):
            lb2 = 1
        if(lb1==lb2):
            Accuracy = Accuracy + 1
            l.append(1)
        else:
            l.append(0)

    return float(float(Accuracy)/float(len(next_label))),l


def figure_faults(test,data,preds):

    l= []

    test_dict = {}
    data_dict = {}

    for row in test.iterrows():
        test_dict[row[1][1]] = 1

    for row in data.iterrows():
        data_dict[row[1][1]] = row[1][(len(row[1])-2)]

    index = 0
    
    for key in sorted(data_dict.keys()):
        v = test_dict.get(key)
        if(v!=None):
            if(preds[index]==1):
                l.append(3)
            else:
                l.append(2)
            index += 1
        else:
            l.append(data_dict[key])

    price_dict = {}
    data2 = pd.read_csv("../Datasets/btc_prices.csv")
    for row in data2.iterrows():
        a = datetime.fromtimestamp(row[1][0])
        aa = a.strftime("%Y-%m-%d")
        price_dict[aa] = row[1][1]

    x = []
    y = []
    for key in sorted(data_dict.keys()):
        v = price_dict.get(key)
        if(v!=None):
            k = time.mktime(time.strptime(key, "%Y-%m-%d"))
            y.append(price_dict[key])
            x.append(k)

    #black for training data 
    #blue for wrong prediction
    #yellow for right prediction
    c = np.array(['black','black','red','yellow'])
            
    fig = plt.figure(figsize=(8,8))
    plt.scatter(x, y, c=l, cmap=matplotlib.colors.ListedColormap(c))
    fig.savefig('Plots/' +'preds_plot.png')

    cb = plt.colorbar()
    loc = np.arange(0,max(l),max(l)/float(len(c)))
    cb.set_ticks(loc)
    cb.set_ticklabels(c)


        

