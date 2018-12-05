import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def convert_data_to_arrays(data):

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

        if(int(row[1][(len(row[1])-2)])==0):
            label = [0,1]
        else:
            label = [1,0]

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


def figure_price(price,asset,label,window):
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
        
            
    fig = plt.figure(figsize=(8,8))
    plt.scatter(x, y, c=label, cmap=matplotlib.colors.ListedColormap(c))
    fig.savefig('Plots/' + asset + '_plot_labels.png')

    cb = plt.colorbar()
    loc = np.arange(0,max(label),max(label)/float(len(c)))
    cb.set_ticks(loc)
    cb.set_ticklabels(c)
