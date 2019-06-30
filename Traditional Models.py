# We will use ARIMA model for prediction.

# AR: Autoregression. A model that uses the dependent relationship between an observation and some number of lagged observations.
# I: Integrated. The use of differencing of raw observations (e.g. subtracting an observation from an observation at the previous time step) in order to make the time series stationary.
# MA: Moving Average. A model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations. ARIMA has three parameters (p,d,q).
# p: The number of lag observations included in the model, also called the lag order.
# d: The number of times that the raw observations are differenced, also called the degree of differencing.
# q: The size of the moving average window, also called the order of moving average.
# Source : https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/

# I will follow you through a method to choose (p,d,q)

def plot_acf_pacf(data,lags=50):
    plot_acf(low_shift_diff,lags=lags)
    plot_pacf(low_shift_diff,lags=lags)
    plt.show() 

# Correlation represents the strength of relationship between 2 variables. We are using Pearson's correlation coefficient to determine correlation. It ranges from -1 to 1.
# 0 signifies no correlation. Autocorrelation is correlation between an observation in time series and a lagged observation in the same time series. 
# Autocorrelation takes into account the direct correlations as well as indirect correlations due to the intervening observations.Partial Correlation function does not take indirect correlations into account and we can know direct correlation between two observations in time series. 
# Source : https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/


plot_acf_pacf(low_shift_diff,lags=10)

# If value of correlation coefficient is above the red band, then it is considered to be significant. The value of p is lag value just after which the autocorrelations becomes insignificant for the first time. 
# The value of q is lag value just after which the partial autocorrelations becomes insignificant for the first time. So, here p=2 and q=2. This is not a hard and fast rule. You can play with values of p and q around 2 and see what changes occur.

def fit(data,d=0,p=2,q=2):
    model=ARIMA(data,(p,d,q))
    model_fit=model.fit(disp=0)
    fitted_values=model_fit.fittedvalues
    score=math.sqrt(mean_squared_error(data,fitted_values))
    return fitted_values,score
        
def plot_values(predictions,data,score):
    plt.figure(figsize=(10,5))
    plt.plot(data,label='original')
    plt.plot(predictions,label='Fitted values',color='black')
    plt.title('RMSE : '+str(score))
    plt.legend(loc='best')
    plt.show()  


# ARIMA Model

fitted_values,score=fit(low_shift_diff)
plot_values(fitted_values,low_shift_diff,score)

# Now we have to convert it in original scale. 
# We can convert so by cumulative addition of fitted_vallues followed by addition of whole series to another series of base number.

def original_scale(fitted_values,data):
    fitted_values_cumsum=fitted_values.cumsum()
    new_series=pd.Series(data[0],index=data.index)
    original_scale_fit=new_series.add(fitted_values_cumsum,fill_value=0)
    return original_scale_fit

original_scale_fit=original_scale(fitted_values,low)
original_score=math.sqrt(mean_squared_error(original_scale_fit,low))
plot_values(original_scale_fit,low,original_score)

#Auto Regression Model (AR)

fitted_values_ar,score_ar=fit(low_shift_diff,p=2,d=0,q=0)
plot_values(fitted_values_ar,low_shift_diff,score_ar)

original_scale_fit_ar=original_scale(fitted_values_ar,low)
original_score_ar=math.sqrt(mean_squared_error(original_scale_fit_ar,low))
plot_values(original_scale_fit_ar,low,original_score_ar)

# Moving Average Model (MA)

fitted_values_ma,score_ma=fit(low_shift_diff,p=0,d=0,q=2)
plot_values(fitted_values_ma,low_shift_diff,score_ma)

original_scale_fit_ma=original_scale(fitted_values_ma,low)
original_score_ma=math.sqrt(mean_squared_error(original_scale_fit_ma,low))
plot_values(original_scale_fit_ma,low,original_score_ma)

# PROPHET 

low_df=pd.DataFrame(low).reset_index()
low_df=low_df.rename(columns={'Date':'ds','Low':'y'})

train_size=0.8
train_index=int(len(low_df)*train_size)
train=low_df.iloc[:train_index,:]
val=low_df.iloc[train_index:,:]
val_X=val.drop('y',axis=1)
val_y=val['y']

# set the uncertainty interval to 95%
prophet_model=Prophet(interval_width=0.95)
prophet_model.fit(train)
pro_predictions=prophet_model.predict(val_X)


pro_rmse_val=math.sqrt(mean_squared_error(val_y,pro_predictions['yhat']))
print('RMSE for Prophet validation : ' + str(pro_rmse_val/len(val_y))) 
 #Normalize with the length of val set

 prophet_model.plot(pro_predictions,uncertainty=True)
plt.show()

prediction_ts=pro_predictions.loc[:,['ds','yhat']].set_index('ds')
val_ts=val.set_index('ds')

plt.figure(figsize=(10,5))
plt.plot(val_ts,label='original')
plt.plot(prediction_ts,label='prediction',color='green')
plt.legend(loc='best')
plt.show()




