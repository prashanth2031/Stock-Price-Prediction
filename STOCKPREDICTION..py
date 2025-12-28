import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use ('fivethirtyeight')
df = yf.download('AAPL', start='2012-01-01', end='2019-12-17')
df.shape
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD ($)',fontsize=18)
plt.show()
data=df.filter(['Close'])
dataset=data.values
training_data_len=math.ceil(len(dataset) * .8)
training_data_len
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)
scaled_data
train_data=scaled_data[0:training_data_len , :]
x_train=[]
y_train=[]
for i in range(60,len(train_data)):
x_train.append(train_data[i-60:i,0])
y_train.append(train_data[i,0])