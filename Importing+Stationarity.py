#import basic modules

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('ggplot')
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose 
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import make_scorer 
import math
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split,GridSearchCV
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM,Dropout,Dense,Input
from keras.models import Sequential,Model
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
import xgboost as xgb
from fbprophet import Prophet
from keras import backend

#Load data

data=pd.read_csv('../input/GOOGL_2006-01-01_to_2018-01-01.csv',parse_dates=['Date'],index_col='Date')
data.head()

data.shape
data.info()

#Plot the variables

plt.subplots(2,2,figsize=(10,10))
plt.subplot(2,2,1)
data['Open'].plot()
plt.title('Open')
plt.subplot(2,2,2)
data.Close.plot()
plt.title('Close')
plt.subplot(2,2,3)
data.High.plot()
plt.title('High')
plt.subplot(2,2,4)
data.Low.plot()
plt.title('Low')
plt.tight_layout()
plt.show()

# I shall create model for data frequency of 1 Month. I have chosen the feature 'LOW'.
# There is no particular reason for choosing frequency of 1 Month and the 'LOW' feature.

low=data['Low'].asfreq('M')
low.dropna(inplace=True)

# Let us check whether the time series is stationary or not. Stationary time series means constant mean and constant standard deviation and 
# autocovariance that does not depend on time. So, we can check whether mean and standard deviation is constant.

#Using rolling mean and rolling standard deviation.
# I am using 'Low' values from previous 7 days (1 week) for rolling mean

def plot_rolling(data,window=7):
    rolling_mean=data.rolling(window).mean()
    rolling_std=data.rolling(window).std()
    plt.figure(figsize=(10,5))
    plt.plot(data,label='original',color='red')
    plt.plot(rolling_mean,label='rolling mean',color='black')
    plt.plot(rolling_std,label='rolling std',color='green')
    plt.legend(loc='best')
    plt.show()

plot_rolling(low)

#Stationarity can also be checked with Augmented Dickey Fuller test.

#Lets check stationarity using Augmented Dickey-Fuller test

def test_stationarity(data):
    result=adfuller(data)
    print('ADF : ' + str(result[0]))
    print('pvalue : ' + str(result[1]))
    print('Number of lags used : ' + str(result[2]))
    print('Number of obs used : ' + str(result[3]))
    print('Critical value at 1% :' + str(result[4]['1%']))
    print('Critical value at 5% :' + str(result[4]['5%']))
    print('Critical value at 10% :' + str(result[4]['10%']))


test_stationarity(low)

# We see that the ADF value is much higher than even Critival value at 10%.
# This is means there is not even 90% chance that this is stationary time series.

# 2 methods to make time series stationary. First, you can subtract the rolling mean of time series from the the time series itself. 
# Second, you can shift the time series and subtract it from the original time series itself.

#Rolling mean subtraction from original time series
window=2
low_rolling=low.rolling(window).mean()
low_rolling_diff=low-low_rolling
low_rolling_diff=low_rolling_diff.dropna()

test_stationarity(low_rolling_diff)
plot_rolling(low_rolling_diff)

#ADF value is lesser than Critical value at 1%. That means we are more than 99% sure that this is stationary time series.
#Even its rolling mean and rolling std are more or less constant

# Subtraction of shifted time series from original time series
shift=1
low_shifted=low.shift(shift)
low_shift_diff=low-low_shifted
low_shift_diff=low_shift_diff.dropna()
test_stationarity(low_shift_diff)
plot_rolling(low_shift_diff)

#Similar result is obtained. You can use any one of the methods, just make sure that the time series thus obtained is stationary.